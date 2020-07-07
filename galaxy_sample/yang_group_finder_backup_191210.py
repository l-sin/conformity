import numpy as np
from itertools import chain
from statistics import mode, StatisticsError

import sys
sys.path.append('/scratch/Documents/Conformity2019/')
import my_functions as mf
import cosmic_distances as cosmo
import two_point_tools as tp
from galaxy_sample import pair_list, halo_mass_model

from joblib import load

import time
from functools import reduce

class yang_group_finder:
    def __init__(self,sample,sky_fraction, az, pol, redshift):        
        self.sample = sample
        self.sky_fraction = sky_fraction
        self.az, self.pol, self.redshift = map( self.sample.data.get, (az,pol,redshift) )
        self.R_comoving = cosmo.comoving_distance( self.redshift )
        
        src='/scratch/Documents/Conformity2019/SLH2020/models/redshift_dependent/'
        self.HaloMassModel = halo_mass_model.RedshiftDependentModel(src=src)
        
        
        self.x, self.y, self.z = self.sphere2cart( self.az, self.pol)
        spacing = (0.1,200)
        g = tp.spherical_grid( points = np.array([ self.pol, self.az, 3e5*self.redshift ]).T,
                               spacing = spacing )

        P = []
        for chunk in g.iter_pairs(chunk=True):
            pair_chunk = pair_list( np.array(chunk) )
            pair_chunk.parent = self.sample

            pair_chunk.separations = pair_chunk.compute_rp(
                                                             az=az,pol=pol,redshift=redshift,
                                                             angular_units='radians',
                                                             output_units='physical'
                                                            )

            cdz = 3e5*np.abs(pair_chunk.parent.data[redshift][pair_chunk.first()] - 
                             pair_chunk.parent.data[redshift][pair_chunk.second()])

            pair_chunk = pair_chunk.select( np.all([ pair_chunk.first()!=pair_chunk.second(),
                                                     pair_chunk.separations<1, cdz<200 ], axis=0) )

            P.append(pair_chunk.pairs)

        P = np.array(list(chain(*P)))

        pairs = pair_list(P)

        #pairs.alpha = np.arccos( self.x[pairs.first()]*self.x[pairs.second()] + 
        #                         self.y[pairs.first()]*self.y[pairs.second()] + 
        #                         self.z[pairs.first()]*self.z[pairs.second()] )
        pairs.dR_comoving = np.abs(np.diff( self.R_comoving[pairs.pairs], axis=1 )).reshape(pairs.count,)

        pairs.parent = self.sample
        pairs.separations = pairs.compute_rp(az=az,pol=pol,redshift=redshift)

        self.pairs = pairs
        
    @staticmethod    
    def sphere2cart(az,pol):
        x = np.cos(az)*np.sin(pol)
        y = np.sin(az)*np.sin(pol)
        z = -np.cos(pol)
        return x,y,z
        
    @staticmethod
    def Sigma(R,r_scale,concentration):
        # Projected density of NFW, with average density of universe already cancelled out
        # Mpc
        """
        def f(x):
            if x < 1:
                g = np.sqrt(1-x**2)
                y = (1/(x**2-1)) * ( 1 - np.log((1+g)/x)/g )
            elif x==1:
                y = 1/3*np.ones(x.shape)
            else:
                g = np.sqrt(x**2-1)
                y = (1/(x**2-1)) * ( 1 - (np.arctan(g)/g) )
            return y
        """
        
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
    def Assign(PossibleGroups,MutualOnly):
        # If a galaxy can be assigned to more than one group according to this criterion,
        # it is only assigned to the one for which PM (R; z) is the largest.
        #Finally, if all members in two groups can be assigned to one group according to the above criterion, 
        #the two groups are merged into a single group.

        # for each galaxy, go through its possible hosts, and assign it to the one for which P_M is highest
        groupIDs = -np.ones(len(PossibleGroups),).astype(int)

        for idx,pgs in PossibleGroups.items():
            if len(pgs)>0:
                Imax = np.argmax( [x[1] for x in pgs] )
                groupIDs[idx] = pgs[Imax][0]

        #Doing this almost guarantees non-convergence.
        current = max(groupIDs)
        for idx,ID in enumerate(groupIDs):
            if ID==-1:
                current+=1
                groupIDs[idx]=current

        #If a group doesn't have any 

        groups = dict()
        for idx,ID in enumerate(groupIDs):
            if ID not in groups:
                groups[ID] = [idx]
            else:
                groups[ID].append(idx)

        # Strangely, some groups don't survive the P_M, but will pick up 
        # Thus, it will not exist in groups, but another galaxy will want to merge with it.
        # Should check if numbers are correct, but if so, then simply ignore these
        f = lambda x,y: set(x).intersection(set(y))
        PotentialMerge =  { ID: reduce(f,
                                       [[pm[0] for pm in PM if pm[0]!=ID and pm[0] in groups] 
                                            for PM in list(map(PossibleGroups.get,members))])
                                for ID,members in groups.items() }

        PairsOfGroups = list(chain(*[tp.all_pairs([k],v) for k,v in PotentialMerge.items() if len(v)>0]))

        if MutualOnly:
            #The description is worded ambiguously. Whether the following is included
            #depends on whether the text is interpreted literally or not
            PairsOfGroups = [ p for p in PairsOfGroups if (p[1],p[0]) in PairsOfGroups ]
            #quadratic, but small enough


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

    
    
    def run(self, l_z, l_p, B=10, MutualOnly=True, max_iter=10):
        def proj_sep_lite( angular_separation, R_comoving):
            """
            converts angular separation on sky to projected separation, assumes a flat cosmology.
            """
            projected_separation = angular_separation*R_comoving #comoving rproj
            return projected_separation
        
        #src = '/scratch/Documents/Conformity2019/SLH2020/models/'
        #Features,HaloMassModel = load(src+'Features'), load(src+'HaloMassModel')
        
        # 1: Run simple group finder
        P = self.link_pairs_fast(l_z,l_p)
        groupIDs,groups = self.merge_links(P)
        iter_results = [(groupIDs,groups)]
        
        Converged,N_iter = False,0
        while not Converged and N_iter<max_iter:
            #X = Features(training_sample,groups) #LogGroupLen,MaxMass,TotMass,Zvar,GroupRedshift
            #TrueMh = HaloMassModel.predict(X)
            
            TrueMh = self.HaloMassModel.predict(self.sample,groups)
            
            Mh = 10**(TrueMh - 14 + np.log10(0.673)) # units of 10**14 h**-1 Msun

            GroupsRedshift = np.array([np.mean(self.redshift[group]) for group in groups.values()])
            GroupComDist = np.array([np.mean(self.R_comoving[group]) for group in groups.values()])
            r180 = (1.26/0.673) * Mh**(1/3) * (1+GroupsRedshift)**(-1) #Mpc
            vel_disp = 397.9 * Mh**0.3214 # units of km/s

            concentration = 10**( 1.02 - 0.109*(TrueMh-12) ) #Maccio 2007
            r_scale = r180/concentration

            PossibleMemberOf = {idx:[] for idx in range(self.sample.count)}

            t = time.time()
            # 4: Update group memberships using tentative halo information
            for i,(ID,members) in enumerate(groups.items()):

                xgc,ygc,zgc = self.sphere2cart( np.mean(self.az[members]), np.mean(self.pol[members]) )

                angular_separation = np.arccos( self.x*xgc + self.y*ygc + self.z*zgc )
                angular_separation[np.isnan(angular_separation)] = 0
                mean_R = (self.R_comoving + GroupComDist[i] )/2

                rp = proj_sep_lite( angular_separation, mean_R )
                dz = self.redshift - GroupsRedshift[i]

                P_M = (67.3/3e5) * self.Sigma(rp,r_scale[i],concentration[i]) * self.p(dz,vel_disp[i],GroupsRedshift[i])

                for idx in np.where(P_M>B)[0]:
                    PossibleMemberOf[idx].append((ID,P_M[idx]))

            newgroupIDs,newgroups = self.Assign(PossibleMemberOf,MutualOnly)
            print( time.time()-t )

            Converged = all( groupIDs==newgroupIDs )


            if sum(map(len,newgroups.values()))!=self.sample.count:
                raise Exception('Fucked up')


            groupIDs,groups = newgroupIDs,newgroups
            iter_results.append((groupIDs,groups))
            N_iter +=1
            # 5: iterate until converged

        if not Converged and N_iter==max_iter:
            print('Warning, max_iter reached, but group memberships not converged.')
        
        return groupIDs,groups
    
    
    
    def average_density(self, R_s=20):

        average_count = mf.smooth1d( self.R_comoving, np.ones( self.sample.count,), R_s, np.sum)

        dR_minus = np.max([ self.R_comoving-(R_s/2),
                            min(self.R_comoving)*np.ones(len(self.R_comoving))],
                          axis=0)
        
        dR_plus = np.min([ max(self.R_comoving)*np.ones(len(self.R_comoving)),
                           self.R_comoving+(R_s/2)],
                         axis=0)

        volume = self.sky_fraction*(4/3)*np.pi*((dR_plus)**3-(dR_minus)**3)

        return average_count/volume
    
    def link_pairs_fast(self,l_p,l_z):
        if not hasattr(self,'pairs'):
            raise Exception('in order to use this feature, init instance with \'init_pairs = True\'')

        """
        alpha_max = l_per/self.R_comoving
        a_max = np.mean(alpha_max[self.pairs.pairs], axis=1)
        dR_comoving_max = np.mean(l_z[self.pairs.pairs], axis=1)

        linked = np.logical_and(np.sin(self.pairs.alpha)<=a_max, self.pairs.dR_comoving<=dR_comoving_max)
        """
        
        L = self.average_density()**(-1/3)
        L = np.mean(L[self.pairs.pairs],axis=1)
        #has a strong redshift dependence due to selection -- is this still valid?
        
        linked = np.logical_and(self.pairs.separations<=l_p*L,
                                self.pairs.dR_comoving<=l_z*L)

        P = self.pairs.select(linked)
        
        return P
    
    
    def merge_links(self,P):
        
        groupIDs = -np.ones(self.sample.count,).astype(int)
        groups = dict()
        currentID = int(0)

        for pair in P.pairs:

            firstID,secondID = groupIDs[pair]

            if firstID==-1 and secondID==-1:
                groupIDs[pair]=currentID
                groups[currentID]=list(pair)
                currentID+=1

            elif firstID!=-1 and secondID==-1:
                groupIDs[pair[1]]=firstID
                groups[firstID].append(pair[1])

            elif firstID==-1 and secondID!=-1:
                groupIDs[pair[0]]=secondID
                groups[secondID].append(pair[0])

            elif firstID==secondID:
                pass

            else:
                groupIDs[ np.array(groups[secondID]) ] = firstID
                groups[firstID].extend(groups[secondID])
                del groups[secondID]

        current = max(groupIDs)

        for idx,ID in enumerate(groupIDs):
            if ID==-1:
                current+=1
                groupIDs[idx]=current
                groups[current] = [idx] 
        
        return groupIDs,groups
    
    
    
class group:
    def __init__(self,gf,ID,members):
        self.parent = gf
        self.id,self.members = ID, members

        self.HaloMass = self.parent.HaloMassModel.predict(self.parent.sample,{self.id:self.members})
        
        Mh = 10**(self.HaloMass - 14 + np.log10(0.673)) #units of 10**14 h**-1 Msun, for convenience
        
        self.az, self.pol = np.mean(self.parent.az[members]), np.mean(self.parent.pol[members])
        self.redshift = np.mean(self.parent.redshift[members])
        self.R_comoving = np.mean(self.parent.R_comoving[members])
        self.r180 = (1.26/0.673) * Mh**(1/3) * (1+self.redshift)**(-1) #Mpc
        self.vel_disp = 397.9 * Mh**0.3214 #units of km/s
        self.concentration = 10**( 1.02 - 0.109*(self.HaloMass-12) ) #Maccio 2007
        self.r_scale = self.r180/self.concentration

    def __repr__(self):
        return '{}: {}'.format(self.id, self.members)
    
    def rp(self):
        def sphere2cart(az,pol):
            x = np.cos(az)*np.sin(pol)
            y = np.sin(az)*np.sin(pol)
            z = -np.cos(pol)
            return x,y,z
        
        def proj_sep_lite( angular_separation, R_comoving):
            """
            converts angular separation on sky to projected separation, assumes a flat cosmology.
            """
            projected_separation = angular_separation*R_comoving #comoving rproj
            return projected_separation
        
        xgc,ygc,zgc = sphere2cart( self.az, self.pol )

        angular_separation = np.arccos( self.parent.x*xgc + 
                                        self.parent.y*ygc + 
                                        self.parent.z*zgc )
        angular_separation[np.isnan(angular_separation)] = 0
        
        mean_R = ( self.parent.R_comoving + self.R_comoving )/2

        rp = proj_sep_lite( angular_separation, mean_R )
        
        return rp
    
    def dz(self):
        return self.parent.redshift - self.redshift
    
    def Sigma(self):
        return self.parent.Sigma( self.rp(), self.r_scale, self.concentration )
        
    def pz(self):
        return self.parent.p( self.dz(), self.vel_disp, self.redshift )
    
    def Pm(self):
        Sigma = self.parent.Sigma( self.rp(), self.r_scale, self.concentration )
        pz = self.parent.p( self.dz(), self.vel_disp, self.redshift )
        
        return (67.3/3e5) * Sigma * pz