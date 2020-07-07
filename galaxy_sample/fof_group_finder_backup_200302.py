import numpy as np
from itertools import chain
from statistics import mode, StatisticsError
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import sys
sys.path.append('/scratch/Documents/Conformity2019/')
import my_functions as mf
import cosmic_distances as cosmo
import two_point_tools as tp
from galaxy_sample import pair_list, halo_mass_model

from joblib import load

import time
from functools import reduce

class fof_group_finder:
    def __init__(self,sample,sky_fraction, az, pol, redshift, stellarmass, spacing, load_pairs=False, pairs_src=None):  
        self.sample = sample
        self.sky_fraction = sky_fraction
        self.az, self.pol, self.redshift, self.stellarmass = map( self.sample.data.get, (az,pol,redshift,stellarmass) )
        self.R_comoving = cosmo.comoving_distance( self.redshift )
        
        self.x, self.y, self.z = mf.sphere2cart( self.az, self.pol)

        self.grid = tp.spherical_grid( points = np.array([ self.pol, self.az, 3e5*self.redshift ]).T,
                                       spacing = spacing )
        
        if load_pairs:
            if pairs_src is None:
                raise Exception('need pairs_src to load pairs')
            
            self.pairs = pair_list(pairs = load(pairs_src+'/pairs'),
                                   separations = load(pairs_src+'/separations')  )
            
        else:

            P = []
            for chunk in self.grid.iter_pairs(chunk=True):
                pair_chunk = pair_list( np.array(chunk) )
                pair_chunk.parent = self.sample

                pair_chunk.separations = pair_chunk.compute_rp(
                                                                 az=az,pol=pol,redshift=redshift,
                                                                 angular_units='radians',
                                                                 output_units='physical'
                                                                )

                cdz = 3e5*np.abs(pair_chunk.parent.data[redshift][pair_chunk.first()] - 
                                 pair_chunk.parent.data[redshift][pair_chunk.second()])

                separation_limit = cosmo.projected_separation(angular_separation = np.array([ spacing[0] ]),
                                                              redshift = np.array([ min(self.redshift) ]),
                                                              angular_units = 'radians',
                                                              output_units = 'physical')[0]

                pair_chunk = pair_chunk.select( np.all([ pair_chunk.first()!=pair_chunk.second(),
                                                         pair_chunk.separations<separation_limit, cdz<spacing[1] ], axis=0) )

                P.append(pair_chunk.pairs)

            P = np.array(list(chain(*P)))

            pairs = pair_list(P)

            pairs.dR_comoving = np.abs(np.diff( self.R_comoving[pairs.pairs], axis=1 )).reshape(pairs.count,)

            pairs.parent = self.sample
            pairs.separations = pairs.compute_rp(az=az,pol=pol,redshift=redshift)

            self.pairs = pairs
        
        
        
    def global_average_density(self):
        count = self.sample.count
        volume = self.sky_fraction*(4/3)*np.pi*((max(self.R_comoving))**3-(min(self.R_comoving))**3)
        return count/volume
        

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
    
    def link_pairs(self,l_p,l_z,density='redshift'):
        
        if density == 'redshift':
            L = self.average_density()**(-1/3)
        elif density == 'global':
            L = self.global_average_density()**(-1/3)*np.ones(self.sample.count,)
            
        L = np.mean(L[self.pairs.pairs],axis=1)
        #has a strong redshift dependence due to selection -- is this still valid?
        #print(max(l_p*L))
        #print(max(l_z*L))
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
                
        #groups = { k:Group(self,k,v) for k,v in groups.items() }
        
        return groupIDs,groups

    

class Group:
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
    
    def fget(self,f,field):
        return f( self.parent.sample.data[field][self.members] )
    
    def rp(self):
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


def YangEvaluate(sample, true_group_ids, obs_group_ids):

    MatchMvir, completeness, contamination, purity = [], [], [], []
    
    TrueGroups = dict()
    for idx,GrNr in enumerate(true_group_ids):
        if GrNr not in TrueGroups:
            TrueGroups[GrNr] = [idx]
        else:
            TrueGroups[GrNr].append(idx)

    ObsGroups = dict()
    for idx,ObsGrNr in enumerate(obs_group_ids):
        if ObsGrNr not in ObsGroups:
            ObsGroups[ObsGrNr] = [idx]
        else:
            ObsGroups[ObsGrNr].append(idx)

    for GrNr,group in ObsGroups.items():
        if len(group)>=2:
            maxmass = np.argmax(sample.data['StellarMass'][group])
            TrueGroupNr = true_group_ids[group][maxmass]

            Match = TrueGroups[TrueGroupNr]

            Nt = len(Match)#total number of members of true group
            Ng = len(group)#total number of members of obs group
            #Ns = sum(1 for gal in group if gal in Match)#Num. of galaxies in obs group which are correctly identified members
            Ns = len(set(Match).intersection(set(group)))
            Ni = sum(1 for gal in group if gal not in Match)#Num. of galaxies in obs group which are not members of true group

            assert sum(1 for gal in group if gal in Match)==Ns
            assert Ng == Ni + Ns
            
            fc = Ns/Nt # completeness
            fi = Ni/Nt # contamination
            fp = Nt/Ng # purity

            MatchMvir.append( sample.data['CentralMvir'][group][maxmass] )
            completeness.append(fc)
            contamination.append(fi)
            purity.append(fp)
    
    return MatchMvir, completeness, contamination, purity



def YangCurves(sample, true_group_ids, obs_group_ids, thresholds, bin_edges):
    
    MatchMvir, completeness, contamination, purity = map(np.array, YangEvaluate(sample, true_group_ids, obs_group_ids) )
    
    labels = zip( bin_edges[:-1], bin_edges[1:] )
    BM = list(chain(  mf.binmembers(MatchMvir,bin_edges)  ))
    
    curve = lambda stat,members: np.array(
                                    [np.mean( stat[members] > t ) for t in thresholds]
                                         )
    
    curves = {label: {'completeness': curve(completeness,m),
                      'contamination': 1 - curve(contamination,m),
                      'purity': curve(purity,m)} for label,(b,m) in zip(labels,BM) }
    
    return curves


def MyEvaluate(sample,true_group_ids, obs_group_ids):

    P = sample.pair_list
    P = P.select(P.separations<1)
    
    truth = np.equal( *(true_group_ids[P.pairs].T) ) #true samehalo
    prediction = np.equal( *(obs_group_ids[P.pairs].T) ) #obs samehalo
    
    accuracy = accuracy_score( truth, prediction )
    
    merge_fraction = np.mean( np.logical_and(prediction, np.logical_not(truth)) )
    frag_fraction = np.mean( np.logical_and( np.logical_not(prediction), truth) )
    
    return accuracy, merge_fraction, frag_fraction