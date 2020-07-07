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
from galaxy_sample import pair_list,multirun_group_finder,halo_mass_model,MyEvaluate

class hybrid_group_finder(multirun_group_finder):
    
    @staticmethod
    def Sigma(R,r_scale,c200):
        # Projected density of NFW, with average density of universe already cancelled out
        # Mpc
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
    def p(dz,vel_disp,group_z):
        
        const = 1/np.sqrt(2*np.pi)
        term1 = 3e5/( vel_disp*(1+group_z) )
        term2 = np.exp( -(3e5*dz)**2 / (2*(vel_disp**2)*(1+group_z)**2) )
        
        return const*term1*term2
    
    def halo_properties(self,groups,Model):
        """
        Tinker prescription
        """
        weights = 10**self.stellarmass
        
        foo = {k:v for k,v in zip(groups.keys(), Model.predict(self.sample,groups))}
        groups = {k:v for k,v in 
                  sorted( groups.items(), key = lambda kv: foo[kv[0]], reverse = True )}

        #logMh = np.array([ Mh for Mh in ObsHaloMass.values()])
        logMh = np.array(Model.predict(self.sample,groups))

        #should be luminosity weighted, I'll use mass
        GroupsRedshift = np.array([np.average(self.redshift[group],weights=weights[group])
                                   for group in groups.values() ])
        GroupComDist = np.array([np.average(self.R_comoving[group],weights=weights[group])
                                 for group in groups.values() ])
        
        rho_crit =  1.5e11 #solar masses Mpc^-3
        G = 4.304e-9 #km^2 s^-2 Mpc Msun^-1
        r200 = ( ((3*np.pi)/(4*200*0.3*rho_crit))*(10**logMh) )**(1/3)
        vel_disp = np.sqrt( (G*(10**logMh))/(2*r200) ) #*(1+GroupsRedshift)
        
        c200 = 10.0*( (10**logMh)/1e14 )**(-0.11) # pow(mass/1.0E+14,-0.11);
        r_scale = r200/c200

        # 4: Update group memberships using tentative halo information
        weighted_mean_az = np.array(list(map( lambda members: np.average(self.az[members],weights=weights[members]),
                                    list(groups.values()) )))
        weighted_mean_pol = np.array(list(map( lambda members: np.average(self.pol[members],weights=weights[members]),
                                     list(groups.values()) )))
        #xgc,ygc,zgc = mf.sphere2cart( weighted_mean_az, weighted_mean_pol )

        return groups, (logMh, GroupsRedshift, GroupComDist, r200, c200, vel_disp, weighted_mean_az, weighted_mean_pol)
    

    def run(self, Params, AM, B=10, max_iter=5):
        def proj_sep_lite( angular_separation, R_comoving):
            """
            converts angular separation on sky to projected separation, assumes a flat cosmology.
            """
            projected_separation = angular_separation*R_comoving #comoving rproj
            return projected_separation
        
        groupIDs,groups = self.multirun(Params)

        iter_results = [[] for i in range(max_iter)]
        iter_results[0].append((groupIDs,groups))
        
        weights = 10**self.stellarmass
        
        # should do this at init.
        # in the meantime, just run 
        #     'gf.grid.neighbour_elems_precompute = {elem:gf.grid.neighbours(elem) for elem in gf.grid.keys()}' 
        # before run()
        # neighbour_elems_precompute = {elem:self.grid.neighbours(elem) for elem in self.grid.keys()}
        
        for N_iter in range(max_iter):
            Converged = [0]
            inner_iter = 0
            #while np.mean(Converged)<0.98 and inner_iter<20:

            # iffy yang bs
            AbundanceMh = AM.predict(self.sample,groups)
            TotalMs = AM.get_features(self.sample,groups,training=False)
            TotalMs = TotalMs[0] if len(TotalMs)==2 else TotalMs
            Model = halo_mass_model.Interpolator(TotalMs,AbundanceMh)
            
            
            while not all(Converged) and inner_iter<20:
            #while not all(Converged) and inner_iter<10:
                groups, (logMh, GroupsRedshift, GroupComDist, r200, c200, 
                         vel_disp, weighted_mean_az, weighted_mean_pol) = self.halo_properties(groups,Model)

                xgc,ygc,zgc = mf.sphere2cart( weighted_mean_az, weighted_mean_pol )
                r_scale = r200/c200
                
                #should encapsulate the following code:
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
                #/encapsulate

                newgroups = dict()
                newgroupIDs = -np.ones(self.sample.count,).astype(int)
                t = time.time()
                for i,(gc,p,a,r) in enumerate(zip( group_centers_IDs, gc_pol_hash,gc_az_hash,gc_rad_hash )):
                    point_hash = (p,a,r) 
                    neighbours = np.array(list(chain(*list(map( self.grid.__getitem__, 
                                                                self.grid.neighbour_elems_precompute[point_hash])))))

                    angular_separation = np.arccos( self.x[neighbours]*xgc[i] +
                                                    self.y[neighbours]*ygc[i] +
                                                    self.z[neighbours]*zgc[i] )
                    angular_separation[np.isnan(angular_separation)] = 0
                    mean_R = (self.R_comoving[neighbours] + GroupComDist[i] )/2

                    rp = angular_separation*mean_R
                    dz = self.redshift[neighbours] - GroupsRedshift[i]

                    P_M = ((67.3/3e5) * 
                            self.Sigma(rp,r_scale[i],c200[i]) * 
                            self.p(dz,vel_disp[i],GroupsRedshift[i]))

                    members = np.array([neighbours[idx] for idx in np.where(P_M>B)[0] if newgroupIDs[neighbours[idx]]==-1])
                    if members.size == 0:
                        continue
                    newgroups[i] = list(members)
                    newgroupIDs[members] = i

                singletons = np.where(newgroupIDs==-1)[0]
                for singleton in singletons:
                    i+=1
                    newgroups[i] = [singleton]
                    newgroupIDs[singleton] = i

                inner_iter+=1
                print(inner_iter)
                print(time.time()-t)
                
                #print('Scores: ',MyEvaluate(self.sample,self.sample.data['FOFCentralGal'],newgroupIDs))
                
                Converged = groupIDs==newgroupIDs
                print(np.mean( Converged ),' percent of group IDs unchanged' )
                print(' ')

                if sum(map(len,newgroups.values()))!=self.sample.count:
                    raise Exception('Total number of galaxies not preserved, something\'s wrong.')

                groupIDs,groups = newgroupIDs,newgroups
                iter_results[N_iter].append((groupIDs,groups))
                
                
            print('Iteration {} complete'.format(N_iter+1))
        
        return iter_results