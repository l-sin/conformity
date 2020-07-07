import numpy as np
from itertools import chain
from statistics import mode, StatisticsError

import sys
sys.path.append('/scratch/Documents/Conformity2019/')
import my_functions as mf
import cosmic_distances as cosmo
import two_point_tools as tp
from galaxy_sample import pair_list

class group_finder:
    def __init__(self,sample,sky_fraction, az, pol, redshift, init_pairs=False):
        def sphere2cart(az,pol):
            x = np.cos(az)*np.sin(pol)
            y = np.sin(az)*np.sin(pol)
            z = -np.cos(pol)
            return x,y,z
        
        self.sample = sample
        self.sky_fraction = sky_fraction
        self.az, self.pol, self.redshift = map( self.sample.data.get, (az,pol,redshift) )
        self.R_comoving = cosmo.comoving_distance( self.redshift )
        
        self.x, self.y, self.z = sphere2cart( self.az, self.pol)
        
        if init_pairs:
            #spacing = (0.05,1000) #4.32180154 Mpc at z=0.02 # why doesn't this work?
            spacing = (0.116,2000)
            g = tp.spherical_grid( points = np.array([ self.pol, self.az, 3e5*self.redshift ]).T,
                                   spacing = spacing )
            
            P = []
            for chunk in g.iter_pairs(chunk=True):
                pair_chunk = pair_list( np.array(chunk) )
                pair_chunk.parent = self.sample

                pair_chunk.separations = pair_chunk.compute_rp(
                                                                 az=az,
                                                                 pol=pol,
                                                                 redshift=redshift,
                                                                 angular_units='radians',
                                                                 output_units='physical'
                                                                )

                cdz = 3e5*np.abs(pair_chunk.parent.data[redshift][pair_chunk.first()] - 
                                 pair_chunk.parent.data[redshift][pair_chunk.second()])

                pair_chunk = pair_chunk.select( np.all([ pair_chunk.first()!=pair_chunk.second(),
                                                         pair_chunk.separations<4, cdz<2000 ], axis=0) )

                P.append(pair_chunk.pairs)
            
            P = np.array(list(chain(*P)))
            
            pairs = pair_list(P)
            
            pairs.alpha = np.arccos( self.x[pairs.first()]*self.x[pairs.second()] + 
                                     self.y[pairs.first()]*self.y[pairs.second()] + 
                                     self.z[pairs.first()]*self.z[pairs.second()] )
            pairs.dR_comoving = np.abs(np.diff( self.R_comoving[pairs.pairs], axis=1 )).reshape(pairs.count,)
            
            pairs.parent = self.sample
            pairs.separations = pairs.compute_rp(az=az,pol=pol,redshift=redshift)
            
            self.pairs = pairs
        
    def run(self,b,r):
        
        if not hasattr(self,'pairs'):
            P = self.link_pairs(b,r)
        else:
            P = self.link_pairs_fast(b,r)
            
        groupIDs,groups = self.merge_links(P)
        
        return groupIDs,groups
    
    def multirun(self,params):
        """
        params: tuple of param, with each param being (filter, b, r)
        filter: boolean function, with filter(group) returning True for groups to keep on that run

        pseudocode for multirun
        initialize 'already-grouped galaxies' with zeros
        for filter,param in params:
            run link_pairs with param
            knock out pairs which contain already-grouped galaxies
            form group from remaining pairs
            collect groups with filter(group)==True, and update already-grouped galaxies
        """
        multirun_groups = []
        grouped = np.zeros(self.sample.count).astype('bool')

        for f,b,r in params:
            P = self.link_pairs_fast(b,r)
            contains_grouped = np.any(grouped[P.pairs],axis=1)
            P = P.select( np.logical_not(contains_grouped) )

            groupIDs,groups = self.merge_links(P)

            for group in filter(f,groups.values()):
                grouped[group] = True
                multirun_groups.append(group)

        groupIDs = -np.ones(self.sample.count,).astype('int')
        multirun_groups = {GrNr:group for GrNr,group in enumerate(multirun_groups)}
        for GrNr,group in multirun_groups.items():
            groupIDs[group] = GrNr

        SingletonIndex = np.arange(self.sample.count,)[groupIDs==-1]
        for gn, idx in zip(range(GrNr+1, GrNr+1+np.sum(groupIDs==-1)), SingletonIndex):
            groupIDs[idx] = gn
            multirun_groups[gn] = [idx]

        return groupIDs,multirun_groups
    
    
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
    
    def link_pairs(self,b,r):
        #linking lengths perpendicular and parallel to the line of sight
        l_per = b/( self.average_density()**(1/3) )
        l_par = l_per*r

        alpha_max = l_per/self.R_comoving
        P = []
        idx = np.arange(self.sample.count)
        
        #switch to grid speedup for large samples
        for gal in idx:

            alpha = np.arccos(self.x[gal]*self.x + self.y[gal]*self.y + self.z[gal]*self.z)
            dR_comoving = abs(self.R_comoving-self.R_comoving[gal])

            a_max = (alpha_max+alpha_max[gal])/2
            dR_comoving_max = (l_par+l_par[gal])/2

            linked = np.logical_and(np.sin(alpha)<=a_max,
                                    dR_comoving<=dR_comoving_max)

            P.append((gal,idx[linked]))

        pairs = list(chain(*[[(g,n) for n in neighs] for g,neighs in P]))
        P = pair_list(np.array(pairs))
        P.separations = np.zeros(P.count,).astype('bool') #syntactical dummy
        P = P.select(P.first()!=P.second())
        
        return P

    
    def link_pairs_fast(self,b,r):
        if not hasattr(self,'pairs'):
            raise Exception('in order to use this feature, init instance with \'init_pairs = True\'')
        
        l_per = b/( self.average_density()**(1/3) )
        l_par = l_per*r
        
        alpha_max = l_per/self.R_comoving
        a_max = np.mean(alpha_max[self.pairs.pairs], axis=1)
        dR_comoving_max = np.mean(l_par[self.pairs.pairs], axis=1)

        linked = np.logical_and(np.sin(self.pairs.alpha)<=a_max, self.pairs.dR_comoving<=dR_comoving_max)

        P = self.pairs.select(linked)
        
        return P
    
    
    def merge_links(self,P):
        
        groupIDs = -np.ones(self.sample.count,)
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
                groupIDs[idx] = current
                groups[current] = [idx] 

        return groupIDs,groups