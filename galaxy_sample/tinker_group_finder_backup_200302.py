import numpy as np
from itertools import chain
from statistics import mode, StatisticsError
from collections import defaultdict
import time
import sys
sys.path.append('/scratch/Documents/Conformity2019/')
import my_functions as mf
import cosmic_distances as cosmo
import two_point_tools as tp
from galaxy_sample import fof_group_finder,pair_list,halo_mass_model,MyEvaluate

class tinker_group_finder(fof_group_finder):
    @staticmethod
    def Sigma(R,r_scale,c200,modifier=1):
        # Projected density of NFW, with average density of universe already cancelled out
        # Mpc
        
        r_scale*=modifier
        
        x = R/r_scale
        f = -np.ones(R.shape)
        
        g_less1 = np.sqrt(1-x[x<1]**2)
        g_gr1 = np.sqrt(x[x>1]**2-1)
        
        f[x<1] = (1/(x[x<1]**2-1)) * ( 1 - np.log( (1+g_less1)/x[x<1] )/g_less1 )
        f[x==1] = 1/3
        f[x>1] = (1/(x[x>1]**2-1)) * ( 1 - (np.arctan(g_gr1)/g_gr1) )
    
        delta_mean = (200/3) * (c200**3/( np.log(1+c200) - c200/(1+c200) ))
        
        return 2*r_scale*delta_mean*f
    
    
    @staticmethod
    def p(dz,vel_disp,group_z,modifier=1):
        
        #const = 1/np.sqrt(2*np.pi)
        #term1 = 3e5/( vel_disp*(1+group_z) )
        #term2 = np.exp( -(3e5*dz)**2 / (2*(vel_disp**2)*(1+group_z)**2) )
        
        vel_disp*=modifier
        
        const = 1/np.sqrt(2*np.pi)
        term1 = 3e5/( vel_disp )
        term2 = np.exp( -(3e5*dz)**2 / (2*(vel_disp**2)) )
        
        return const*term1*term2
    
    
    def halo_properties(self,groups,Model):
        """
        Takes in groups in dict form, and returns them, as well as group properties, sorted by predicted halo mass        
        """        
        weights = 10**self.stellarmass
        group_centers = dict()
        
        #sort groups into descending halo mass
        foo = {k:v for k,v in zip(groups.keys(), Model.predict(self.sample,groups))}
        sorted_groups = {k:v for k,v in 
                  sorted( groups.items(), key = lambda kv: foo[kv[0]], reverse = True )}
        
        group_centers['logMh'] = np.array(Model.predict(self.sample,sorted_groups))
        
        group_centers['redshift'] = np.array([np.average(self.redshift[group],weights=weights[group])
                                              for group in sorted_groups.values() ])
        group_centers['ComDist'] = np.array([np.average(self.R_comoving[group],weights=weights[group])
                                             for group in sorted_groups.values() ])
        
        rho_crit =  1.5e11 #solar masses Mpc^-3
        G = 4.304e-9 #km^2 s^-2 Mpc Msun^-1
        group_centers['r200'] = ( ((3*np.pi)/(4*200*0.3*rho_crit))*(10**group_centers['logMh']) )**(1/3)
        
        group_centers['vel_disp'] = np.sqrt( (G*(10**group_centers['logMh']))/(2*group_centers['r200']) ) #*(1+GroupsRedshift)
        
        group_centers['c200'] = 10.0*( (10**group_centers['logMh'])/1e14 )**(-0.11) # pow(mass/1.0E+14,-0.11);
        group_centers['r_scale'] = group_centers['r200']/group_centers['c200']

        # 4: Update group memberships using tentative halo information
        group_centers['weighted_mean_az'] = np.array(list(map( lambda members: np.average(self.az[members],
                                                                                          weights=weights[members]),
                                                               list(sorted_groups.values()) )))
        group_centers['weighted_mean_pol'] = np.array(list(map( lambda members: np.average(self.pol[members],
                                                                                           weights=weights[members]),
                                                                list(sorted_groups.values()) )))
        return sorted_groups,group_centers
    
    
    def hash_to_grid(self,group_centers):
        group_centers_grid = defaultdict(list)
        
        gc_rad_hash = np.int32( mf.bin_hash( 3e5*group_centers['redshift'], self.grid.radial_grid ))
        gc_pol_hash = np.int32( mf.bin_hash( group_centers['weighted_mean_pol'], self.grid.polar_grid) )

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
                gc_az_hash[sel] = np.int32( mf.bin_hash( group_centers['weighted_mean_az'][sel],azimuthal_grid) )

        for index,point_hash in enumerate(zip( gc_pol_hash,gc_az_hash,gc_rad_hash )):
            group_centers_grid[point_hash].append(index)

        group_centers_grid.default_factory = None
        
        return gc_pol_hash, gc_az_hash, gc_rad_hash
        
    
    
    def assign(self,sorted_groups,group_centers,B,rscale_mod=1,vel_disp_mod=1):
        
        xgc,ygc,zgc = mf.sphere2cart( group_centers['weighted_mean_az'], group_centers['weighted_mean_pol'] )
        gc_pol_hash, gc_az_hash, gc_rad_hash = self.hash_to_grid(group_centers)
        
        newgroups = dict()
        newgroupIDs = -np.ones(self.sample.count,).astype(int)
        
        for i,(k,p,a,r) in enumerate(zip( sorted_groups.keys(), gc_pol_hash, gc_az_hash, gc_rad_hash )):
            point_hash = (p,a,r)
        
            try:
                neighbours = np.array(list(chain(*list(map( self.grid.__getitem__, 
                                                            self.grid.neighbour_elems_precompute[point_hash])))))
            except:
                neighbours = np.array(list(chain(*list(map( self.grid.__getitem__, 
                                                            self.grid.neighbours(point_hash))))))
        
            angular_separation = np.arccos( self.x[neighbours]*xgc[i] +
                                            self.y[neighbours]*ygc[i] +
                                            self.z[neighbours]*zgc[i] )
            angular_separation[np.isnan(angular_separation)] = 0
            mean_R = (self.R_comoving[neighbours] + group_centers['ComDist'][i] )/2
        
            rp = angular_separation*mean_R
            dz = self.redshift[neighbours] - group_centers['redshift'][i]
        
            P_M = ((67.3/3e5) * 
                    self.Sigma(rp,group_centers['r_scale'][i],group_centers['c200'][i],modifier=rscale_mod) * 
                    self.p(dz,group_centers['vel_disp'][i],group_centers['redshift'][i],modifier=vel_disp_mod))
        
            members = np.array([neighbours[idx] for idx in np.where(P_M>B)[0] if newgroupIDs[neighbours[idx]]==-1])
            if members.size == 0:
                continue
            newgroups[k] = list(members)
            newgroupIDs[members] = k
        
        singletons = np.where(newgroupIDs==-1)[0]
        current = max(newgroups.keys())
        for singleton in singletons:
            current+=1
            newgroups[current] = [singleton]
            newgroupIDs[singleton] = current
            
        return newgroupIDs, newgroups
    

    def run(self, B, rscale_mod=1, vel_disp_mod=1, max_iter=30, masslimited=False):
        """
        the masslimited argument is only used in testing on a mock volume with a flat mass cut at logMstar=9.0,
        and should otherwise always be False
        """
        #rscaleMod,veldispMod,B = 0.6, 1.7, 17
        
        if not hasattr(self.sample,'pairs'):
            self.sample.define_pairs(self.pairs)

        groupIDs = np.arange(self.sample.count)
        groups = {i:[i] for i in range(self.sample.count)}

        iter_results = [(groupIDs,groups)]
        
        #this part should not be hardcoded
        AMproto = (halo_mass_model.RedshiftDependentAbundanceMatching if not masslimited 
                     else halo_mass_model.AbundanceMatching)
        src = ('/scratch/Documents/Conformity2019/SLH2020/models/z_abundance_matching/' if not masslimited
               else '/scratch/Documents/Conformity2019/SLH2020/models/abundance_matching/')
        
        Converged = [0]
        N_iter = 0

        while not all(Converged) and N_iter<max_iter:
            # On first iteration, initialize haloes via SHAM.
            # Thereafter, compute halo properties via normal AM

            SHAM = True if N_iter==0 else False
            AM = AMproto(src=src,SHAM=SHAM)

            AbundanceMh = AM.predict(self.sample,groups)
            TotalMs = AM.get_features(self.sample,groups,training=False)
            TotalMs = TotalMs[0] if len(TotalMs)==2 else TotalMs
            Model = halo_mass_model.Interpolator(TotalMs,AbundanceMh)
                
            sorted_groups,group_centers = self.halo_properties(groups,Model)
            
            t = time.time()
            newgroupIDs,newgroups = self.assign(sorted_groups,group_centers,B,rscale_mod,vel_disp_mod)
            print(time.time()-t)
            
            N_iter+=1
            print(N_iter)
            
            print(MyEvaluate(self.sample, self.sample.data['FOFCentralGal'], newgroupIDs))

            Converged = groupIDs==newgroupIDs
            print(np.mean( Converged ),' percent of group IDs unchanged' )
            print(' ')

            if sum(map(len,newgroups.values()))!=self.sample.count:
                raise Exception('Total number of galaxies not preserved, something\'s wrong.')

            groupIDs,groups = newgroupIDs,newgroups
            iter_results.append((groupIDs,groups))
        
        return iter_results