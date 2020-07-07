import numpy as np
import h5py
import sys
sys.path.insert(0,'/scratch/Documents/Conformity2019/')
import my_functions as mf
import two_point_tools as tp
from galaxy_sample import galaxy_sample,pair_list,halo_mass_model,tinker_group_finder,MyEvaluate
import time
from joblib import dump, load
from sklearn import metrics

BAS = lambda x: np.sqrt( x[0]**2 - (x[1]**2+ x[2]**2) )

c,h = 299792.458, 0.677
AM_redshift_bins = np.arange(0.02,0.09,0.01)
AM_src = '../data/LGalaxies/'

N_iter = int(sys.argv[1])
#print(N_iter)

data_sources = ['../data/LGalaxies/{}/'.format(i) for i in range(8)]

training_src = data_sources[0]

data = load(training_src+'data')
part = galaxy_sample( data )
part.define_pairs( pair_list( load(training_src+'pairs'), load(training_src+'separations'), parent=part) )

#initialize group finder
try:
    AM = halo_mass_model.RedshiftDependentAbundanceMatching(src=AM_src,SHAM=False,redshift_bins=AM_redshift_bins)
except:
    # easiest way to handle this is to have run derive_quantities.py once with a dummy set of group finder params
    raise Exception('Need to have halo mass function and subhalo mass function defined already under '+AM_src)

gf = tinker_group_finder(
                         part,
                         sky_fraction=None,
                         az='azimuthal_angle',
                         pol='polar_angle',
                         redshift='redshift',
                         stellarmass='StellarMass',
                         spacing=(0.04,2000),
                         load_pairs=True,
                         pairs_src=training_src
                         )
gf.grid.neighbour_elems_precompute = {elem:gf.grid.neighbours(elem) for elem in gf.grid.keys()}


#training
B_range = np.random.uniform(low=1,high=100,size=N_iter)
rm_range = np.random.uniform(low=0.1,high=10,size=N_iter)
vm_range = np.random.uniform(low=0.1,high=10,size=N_iter)
    
optimization_results = []
start_time = time.time()
for B, rscaleMod, veldispMod in zip(B_range, rm_range, vm_range):
    iter_results = gf.run(AM, B, rscaleMod, veldispMod, max_iter=5, verbose=False)
    groupIDs,_ = iter_results[-1]
    
    optimization_results.append(
        (
        (B, rscaleMod, veldispMod),
        BAS(MyEvaluate(part, part.data['FOFCentralGal'], groupIDs)),
        metrics.adjusted_rand_score(part.data['FOFCentralGal'], groupIDs),
        metrics.fowlkes_mallows_score(part.data['FOFCentralGal'], groupIDs)
        )
    )
print('Ran {} iterations in {} seconds.'.format(N_iter,time.time()-start_time))
dump( optimization_results, '../data/LGalaxies/optimization_results_{}iters'.format(N_iter) )