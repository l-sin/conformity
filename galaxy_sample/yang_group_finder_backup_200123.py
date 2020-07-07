import numpy as np
from itertools import chain
from statistics import mode, StatisticsError

import sys
sys.path.append('/scratch/Documents/Conformity2019/')
import my_functions as mf
import cosmic_distances as cosmo
import two_point_tools as tp
from galaxy_sample import pair_list, halo_mass_model, fof_group_finder

from joblib import load

import time
from functools import reduce

class yang_group_finder(fof_group_finder):

    @staticmethod
    def Sigma(R,r_scale,concentration):
        # Projected density of NFW, with average density of universe already cancelled out
        # Mpc
        x = R/r_scale
        f = -np.ones(R.shape)
        
        g_less1 = np.sqrt(1-x[x<1]**2)
        g_gr1 = np.sqrt(x[x>1]**2-1)
        
        f[x<1] = (1/(x[x<1]**2-1)) * ( 1 - np.log( (1+g_less1)/x[x<1] )/g_less1 )
        f[x==1] = 1/3
        f[x>1] = (1/(x[x>1]**2-1)) * ( 1 - (np.arctan(g_gr1)/g_gr1) )
    
        c180 = concentration
        delta_mean = (180/3) * (c180**3/( np.log(1+c180) - c180/(1+c180) ))
        
        #return 2*r_scale*delta_mean*f(R/r_scale)
        return 2*r_scale*delta_mean*f
    
    @staticmethod
    def p(dz,vel_disp,group_z):
        
        const = 1/np.sqrt(2*np.pi)
        term1 = 3e5/( vel_disp*(1+group_z) )
        term2 = np.exp( -(3e5*dz)**2 / (2*vel_disp**2*(1+group_z)**2) )
        
        return const*term1*term2
    
    @staticmethod
    def Assign(PossibleGroups):
        # PossibleGroups = {gal_idx:[(groupid,PM),(groupid,PM),(groupid,PM)...]}

        # If a galaxy can be assigned to more than one group according to this criterion,
        # it is only assigned to the one for which PM (R; z) is the largest.
        groupIDs = -np.ones(len(PossibleGroups),).astype(int)

        for idx,pgs in PossibleGroups.items():
            #pgs = [(groupid,PM),(groupid,PM),(groupid,PM)...]
            if len(pgs)>0:
                Imax = np.argmax( [x[1] for x in pgs] )
                groupIDs[idx] = pgs[Imax][0]

        current = max(groupIDs)
        for idx,ID in enumerate(groupIDs):
            if ID==-1:
                current+=1
                groupIDs[idx]=current

        groups = dict()
        for idx,ID in enumerate(groupIDs):
            if ID not in groups:
                groups[ID] = [idx]
            else:
                groups[ID].append(idx)
                

        #Finally, if all members in two groups can be assigned to one group according to the above criterion, 
        #the two groups are merged into a single group.
        #Phrasing of Campbell: ... if all the members of one group can be assigned to another, the groups are merged.
        

        f = lambda x,y: set(x).intersection(set(y))
        PotentialMerge=dict()
        for ID,members in groups.items():
            # PossibleGroups = {gal_idx:[(groupid,PM),(groupid,PM),(groupid,PM)...]}
            PossibleGroupsOfMembers = list(map(PossibleGroups.get,members))
            
            #PossibleGroupsOfMembers = [[(groupid,PM),(groupid,PM),(groupid,PM)...],
            #                           [(groupid,PM),(groupid,PM),(groupid,PM)...],
            #                           [(groupid,PM),(groupid,PM),(groupid,PM)...],
            #                           ...]
            # Strangely, some groups don't survive the P_M, but will pick up a new member
            # Thus, it will not exist in groups, but another galaxy will want to merge with it.
            # Should check if numbers are correct, but if so, then simply ignore these
            

            PossibleGroupsOfMembers = [[x[0] for x in PG if x[0]!=ID and x[0] in groups] for PG in PossibleGroupsOfMembers]
            
            #The texts do not specify what to do if all the members of one group can be assigned to more than one other group.
            
            
            PotentialMerge[ID] = reduce(f,PossibleGroupsOfMembers)
            #Assuming that 'bridged' merging is allowed, then do nothing
            #Assuming that 'bridged' merging is not allowed, then assign to the highest PM
            if len(PotentialMerge[ID])>1:
                currentMaxPM,maxPMcand = 0,0
                for cand in PotentialMerge[ID]:
                    PossibleGroupsOfMembers = list(map(PossibleGroups.get,members))
                    meanPM = np.mean([ [x[1] for x in PG if x[0]==cand] for PG in PossibleGroupsOfMembers])
                    if meanPM>currentMaxPM:
                        currentMaxPM,maxPMcand = meanPM,cand
                PotentialMerge[ID] = [maxPMcand]
                    
        PairsOfGroups = list(chain(*[tp.all_pairs([k],v) for k,v in PotentialMerge.items() if len(v)>0]))

        SupergroupIDs = {k:-1 for k in np.unique(np.array(PairsOfGroups))}
        Supergroups = dict()
        currentID = int(0)

        for pair in PairsOfGroups:

            firstID,secondID = map(SupergroupIDs.get,pair)

            if firstID==-1 and secondID==-1:
                SupergroupIDs[pair[0]]=currentID
                SupergroupIDs[pair[1]]=currentID
                Supergroups[currentID]=list(pair)
                currentID+=1

            elif firstID!=-1 and secondID==-1:
                SupergroupIDs[pair[1]]=firstID
                Supergroups[firstID].append(pair[1])

            elif firstID==-1 and secondID!=-1:
                SupergroupIDs[pair[0]]=secondID
                Supergroups[secondID].append(pair[0])

            elif firstID==secondID:
                pass

            else:
                for grp in Supergroups[secondID]:
                    SupergroupIDs[ grp ] = firstID
                Supergroups[firstID].extend(Supergroups[secondID])
                del Supergroups[secondID]
        #return groups,groupIDs,PotentialMerge,SupergroupIDs,Supergroups

        # Merge members of each supergroup

        #The following is incorrect?

        for Supergroup in Supergroups.values():
            for mergee in Supergroup[1:]:
                groups[Supergroup[0]].extend( groups[mergee] )
                groupIDs[ groups[mergee] ] = Supergroup[0]

                groups.pop(mergee)

        return groupIDs,groups

    
    def run(self, l_z, l_p, B=10, max_iter=10):
        def proj_sep_lite( angular_separation, R_comoving):
            """
            converts angular separation on sky to projected separation, assumes a flat cosmology.
            """
            projected_separation = angular_separation*R_comoving #comoving rproj
            return projected_separation
        
        #src = '/scratch/Documents/Conformity2019/SLH2020/models/'
        #Features,HaloMassModel = load(src+'Features'), load(src+'HaloMassModel')
        
        # 1: Run simple group finder
        P = self.link_pairs(l_z,l_p)
        groupIDs,groups = self.merge_links(P)
        iter_results = [(groupIDs,groups)]
        
        weights = 10**self.stellarmass
        
        Converged,N_iter = False,0
        while not Converged and N_iter<max_iter:
            #X = Features(training_sample,groups) #LogGroupLen,MaxMass,TotMass,Zvar,GroupRedshift
            #TrueMh = HaloMassModel.predict(X)
            
            TrueMh = self.HaloMassModel.predict(self.sample,groups)
            
            Mh = 10**(TrueMh - 14 + np.log10(0.673)) # units of 10**14 h**-1 Msun

            #GroupsRedshift = np.array([np.mean(self.redshift[group]) for group in groups.values()])
            #GroupComDist = np.array([np.mean(self.R_comoving[group]) for group in groups.values()])
            
            #should be luminosity weighted, I'll use mass
            GroupsRedshift = np.array([ np.average(self.redshift[group],weights=weights[group]) for group in groups.values() ])
            GroupComDist = np.array([ np.average(self.R_comoving[group],weights=weights[group]) for group in groups.values() ])
            
            r180 = (1.26/0.673) * Mh**(1/3) * (1+GroupsRedshift)**(-1) #Mpc
            vel_disp = 397.9 * Mh**0.3214 # units of km/s

            concentration = 10**( 1.02 - 0.109*(TrueMh-12) ) #Maccio 2007
            r_scale = r180/concentration

            PossibleMemberOf = {idx:[] for idx in range(self.sample.count)}

            t = time.time()
            # 4: Update group memberships using tentative halo information
            for i,(ID,members) in enumerate(groups.items()):
                #print(np.average(self.az[members],weights=weights[members]))
                #print(np.average(self.pol[members],weights=weights[members]))
                xgc,ygc,zgc = mf.sphere2cart( np.array([np.average(self.az[members],weights=weights[members])]), 
                                              np.array([np.average(self.pol[members],weights=weights[members])]) )

                angular_separation = np.arccos( self.x*xgc + self.y*ygc + self.z*zgc )
                angular_separation[np.isnan(angular_separation)] = 0
                mean_R = (self.R_comoving + GroupComDist[i] )/2

                rp = proj_sep_lite( angular_separation, mean_R )
                dz = self.redshift - GroupsRedshift[i]

                P_M = (67.3/3e5) * self.Sigma(rp,r_scale[i],concentration[i]) * self.p(dz,vel_disp[i],GroupsRedshift[i])

                for idx in np.where(P_M>B)[0]:
                    PossibleMemberOf[idx].append((ID,P_M[idx]))

            newgroupIDs,newgroups = self.Assign(PossibleMemberOf)
            print( time.time()-t )

            Converged = all( groupIDs==newgroupIDs )


            if sum(map(len,newgroups.values()))!=self.sample.count:
                raise Exception('Total number of galaxies not preserved, something\'s wrong.')


            groupIDs,groups = newgroupIDs,newgroups
            iter_results.append((groupIDs,groups))
            N_iter +=1
            # 5: iterate until converged

        if not Converged and N_iter==max_iter:
            print('Warning, max_iter reached, but group memberships not converged.')
        
        return groupIDs,groups
    
    
    
#class group:
#    def __init__(self,gf,ID,members):
#        self.parent = gf
#        self.id,self.members = ID, members
#
#        self.HaloMass = self.parent.HaloMassModel.predict(self.parent.sample,{self.id:self.members})
#        
#        Mh = 10**(self.HaloMass - 14 + np.log10(0.673)) #units of 10**14 h**-1 Msun, for convenience
#        
#        self.az, self.pol = np.mean(self.parent.az[members]), np.mean(self.parent.pol[members])
#        self.redshift = np.mean(self.parent.redshift[members])
#        self.R_comoving = np.mean(self.parent.R_comoving[members])
#        self.r180 = (1.26/0.673) * Mh**(1/3) * (1+self.redshift)**(-1) #Mpc
#        self.vel_disp = 397.9 * Mh**0.3214 #units of km/s
#        self.concentration = 10**( 1.02 - 0.109*(self.HaloMass-12) ) #Maccio 2007
#        self.r_scale = self.r180/self.concentration
#
#    def __repr__(self):
#        return '{}: {}'.format(self.id, self.members)
#    
#    def rp(self):
#        def proj_sep_lite( angular_separation, R_comoving):
#            """
#            converts angular separation on sky to projected separation, assumes a flat cosmology.
#            """
#            projected_separation = angular_separation*R_comoving #comoving rproj
#            return projected_separation
#        
#        xgc,ygc,zgc = sphere2cart( self.az, self.pol )
#
#        angular_separation = np.arccos( self.parent.x*xgc + 
#                                        self.parent.y*ygc + 
#                                        self.parent.z*zgc )
#        angular_separation[np.isnan(angular_separation)] = 0
#        
#        mean_R = ( self.parent.R_comoving + self.R_comoving )/2
#
#        rp = proj_sep_lite( angular_separation, mean_R )
#        
#        return rp
#    
#    def dz(self):
#        return self.parent.redshift - self.redshift
#    
#    def Sigma(self):
#        return self.parent.Sigma( self.rp(), self.r_scale, self.concentration )
#        
#    def pz(self):
#        return self.parent.p( self.dz(), self.vel_disp, self.redshift )
#    
#    def Pm(self):
#        Sigma = self.parent.Sigma( self.rp(), self.r_scale, self.concentration )
#        pz = self.parent.p( self.dz(), self.vel_disp, self.redshift )
#        
#        return (67.3/3e5) * Sigma * pz