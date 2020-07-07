import numpy as np
from itertools import chain
from statistics import mode, StatisticsError

import sys
sys.path.append('/scratch/Documents/Conformity2019/')
import my_functions as mf
import cosmic_distances as cosmo
import two_point_tools as tp
from galaxy_sample import pair_list, halo_mass_model, fof_group_finder, MyEvaluate
from collections import defaultdict

from sklearn.linear_model import LinearRegression

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
        term2 = np.exp( -(3e5*dz)**2 / (2*(vel_disp**2)*(1+group_z)**2) )
        
        return const*term1*term2
    
    
    @staticmethod
    def ResolveMergers(PossibleGroups,groups):
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
                
        return PotentialMerge
    
    
    @staticmethod
    def Assign(ResolveMergers,PossibleGroups):
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
                

        PotentialMerge = ResolveMergers(PossibleGroups,groups)
                    
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

    
    def run(self, l_z, l_p, mhalo_model, B=10, max_iter=5):
        def proj_sep_lite( angular_separation, R_comoving):
            """
            converts angular separation on sky to projected separation, assumes a flat cosmology.
            """
            projected_separation = angular_separation*R_comoving #comoving rproj
            return projected_separation
        
        # 1: Run simple group finder
        #P = self.link_pairs(l_z,l_p,density='redshift')
        P = self.link_pairs(l_z,l_p,density='global')
        groupIDs,groups = self.merge_links(P)
        iter_results = [[] for i in range(max_iter)]
        iter_results[0].append((groupIDs,groups))
        
        weights = 10**self.stellarmass
        
        for N_iter in range(max_iter):
            #define Mh-M* relation
            if N_iter==0:
                TotalMs = halo_mass_model.SimpleModel(load_params=False).get_features(self.sample,groups)
                features = np.array([TotalMs,TotalMs**2]).T.reshape(-1,2)
                
                Model = halo_mass_model.SimpleModel(load_params=False)#.fit( np.array([Ms,Ms**2]).T.reshape(-1,2), Mh )
                Model.Model.intercept_ = 25.020301549565737
                Model.Model.coef_ = np.array([-3.34185887,  0.20127182])
            else:
                #assign halo masses to groups via abundance matching
                
                #src = '/scratch/Documents/Conformity2019/SLH2020/models/abundance_matching/complete/'
                #AM = halo_mass_model.AbundanceMatching(src=src)
                
                #src='/scratch/Documents/Conformity2019/SLH2020/models/z_abundance_matching/'
                #AM = halo_mass_model.RedshiftDependentAbundanceMatching(src=src)
                
                AbundanceMh = mhalo_model.predict(self.sample,groups)
                TotalMs = mhalo_model.get_features(self.sample,groups,training=False)
                
                if len(TotalMs)==2:
                    TotalMs = TotalMs[0]
                
                #define Mh-M* relation 
                Model = halo_mass_model.Interpolator(TotalMs,AbundanceMh)
                
            Converged = False
            inner_iter = 0
            while not Converged and inner_iter<20:
                # For the current Mh-M* relation, iterate until group memberships are converged
                TrueMh = Model.predict(self.sample,groups)
                Mh = 10**(TrueMh - 14 + np.log10(0.673)) # units of 10**14 h**-1 Msun

                #should be luminosity weighted, I'll use mass
                GroupsRedshift = np.array([np.average(self.redshift[group],weights=weights[group]) 
                                           for group in groups.values() ])
                GroupComDist = np.array([np.average(self.R_comoving[group],weights=weights[group]) 
                                         for group in groups.values() ])

                r180 = (1.26/0.673) * Mh**(1/3) * (1+GroupsRedshift)**(-1) #Mpc
                vel_disp = 397.9 * Mh**0.3214 #units of km/s

                #concentration = 10**( 1.02 - 0.109*(TrueMh-12) ) #Maccio 2007
                concentration = 10**( 1.071 - 0.098*(TrueMh-12) ) #Maccio 2007
                r_scale = r180/concentration

                PossibleMemberOf = {idx:[] for idx in range(self.sample.count)}

                # 4: Update group memberships using tentative halo information
                weighted_mean_az = np.array(list(map( lambda members: np.average(self.az[members],weights=weights[members]),
                                            list(groups.values()) )))
                weighted_mean_pol = np.array(list(map( lambda members: np.average(self.pol[members],weights=weights[members]),
                                             list(groups.values()) )))
                xgc,ygc,zgc = mf.sphere2cart( weighted_mean_az, weighted_mean_pol )

                group_centers_IDs = list(groups.keys())
                group_centers_grid = defaultdict(list)
                gc_rad_hash = np.int32( mf.bin_hash(3e5*GroupsRedshift, self.grid.radial_grid ))
                gc_pol_hash = np.int32( mf.bin_hash(weighted_mean_pol,self.grid.polar_grid) )

                #compute delta_az (azimuthal coordinate distance) corresponding to 'spacing'
                delta_az = lambda spacing,phi: np.arccos(( np.cos(spacing)-np.cos(phi)**2 )/( np.sin(phi)**2 ) )

                gc_az_hash = -np.ones(gc_pol_hash.shape).astype('int')
                for p,phi_range in enumerate( zip(self.grid.polar_grid[:-1],self.grid.polar_grid[1:]) ):

                    sel=gc_pol_hash==p

                    if p==0 or p==len(self.grid.polar_grid)-2:
                        gc_az_hash[sel] = 0
                    else:
                        phi = phi_range[np.argmin( np.sin(phi_range) )]
                        azimuthal_spacing = (2*np.pi)/np.floor((2*np.pi)/delta_az(self.grid.angular_spacing,phi))
                        azimuthal_grid = np.arange( 0, 2*np.pi+azimuthal_spacing, azimuthal_spacing )
                        gc_az_hash[sel] = np.int32( mf.bin_hash(weighted_mean_az[sel],azimuthal_grid) )

                for index,point_hash in enumerate(zip( gc_pol_hash,gc_az_hash,gc_rad_hash )):
                    group_centers_grid[point_hash].append(index)

                group_centers_grid.default_factory = None

                t = time.time()
                for elem,gcs in group_centers_grid.items():                
                    neighbours = np.array(list(chain(*list(map( self.grid.__getitem__, self.grid.neighbours(elem))))))
                    for gc in gcs:
                        angular_separation = np.arccos( self.x[neighbours]*xgc[gc] +
                                                        self.y[neighbours]*ygc[gc] +
                                                        self.z[neighbours]*zgc[gc] )
                        angular_separation[np.isnan(angular_separation)] = 0
                        mean_R = (self.R_comoving[neighbours] + GroupComDist[gc] )/2

                        rp = angular_separation*mean_R
                        dz = self.redshift[neighbours] - GroupsRedshift[gc]

                        P_M = ((67.3/3e5) * 
                                self.Sigma(rp,r_scale[gc],concentration[gc]) * 
                                self.p(dz,vel_disp[gc],GroupsRedshift[gc]))

                        for idx in np.where(P_M>B)[0]:
                            PossibleMemberOf[neighbours[idx]].append((group_centers_IDs[gc],P_M[idx]))

                print('Inner iteration number ',inner_iter+1)
                print(time.time()-t,' seconds')
                
                newgroupIDs,newgroups = self.Assign(self.ResolveMergers,PossibleMemberOf)
                print('Scores: ',MyEvaluate(self.sample,self.sample.data['FOFCentralGal'],newgroupIDs))
                print(np.mean( groupIDs==newgroupIDs ),' percent of group IDs unchanged' )
                
                Converged = all( groupIDs==newgroupIDs )
                print(' ')

                if sum(map(len,newgroups.values()))!=self.sample.count:
                    raise Exception('Total number of galaxies not preserved, something\'s wrong.')

                groupIDs,groups = newgroupIDs,newgroups
                iter_results[N_iter].append((groupIDs,groups))
                inner_iter+=1
                
            print('Iteration {} complete'.format(N_iter+1))
        
        return iter_results
    
    
    
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