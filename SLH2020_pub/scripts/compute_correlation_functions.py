import numpy as np
import sys
sys.path.insert(0,'/scratch/Documents/Conformity2019/')
import my_functions as mf
import two_point_tools as tp
from galaxy_sample import galaxy_sample,pair_list
from joblib import load,dump

sample = sys.argv[1]
if sample=='sdss':
    data_sources = ['../data/SDSS/']
    
elif sample=='mock':
    data_sources = ['../data/LGalaxies/{}/'.format(i) for i in range(8)]


for data_src in data_sources:
    data = load(data_src+'data')
    pairs = load(data_src+'pairs')
    separations = load(data_src+'separations')

    part = galaxy_sample(data)
    part.define_pairs( pair_list(pairs,separations,parent=part) )

    part.define_corrfunc('full_sample',var='Delta_md')
    obs_same_halo = np.equal(*(part.pair_list.get('ObsGrNr').T))
    obs_diff_halo = np.logical_not(obs_same_halo)
    
    if sample=='sdss':
        selections = { 'obs_same_halo': obs_same_halo,
                       'obs_diff_halo': obs_diff_halo}
    elif sample=='mock':
        true_same_halo = np.equal(*(part.pair_list.get('FOFCentralGal').T))
        true_diff_halo = np.logical_not(true_same_halo)
        
        selections = {'true_same_halo':true_same_halo,
                      'true_diff_halo':true_diff_halo,
                      'obs_same_halo': obs_same_halo,
                      'obs_diff_halo': obs_diff_halo,
                      'same_correct':np.logical_and( obs_same_halo, true_same_halo ),
                      'merge':np.logical_and( obs_same_halo, true_diff_halo ),
                      'diff_correct':np.logical_and( obs_diff_halo, true_diff_halo ),
                      'frag':np.logical_and( obs_diff_halo, true_same_halo ) }

    for name,selection in selections.items():
        part.define_corrfunc(name,var='Delta_md',pairs=selection)
        part.define_corrfunc(name+'_unwtd',var='Delta_md',pairs=selection,weighted=False)

    pickle = {name:{
                'results':cf.results,
                'errorbars':cf.errorbars,
                'stdev':cf.stdev,
                'selection':cf.selection,
                'config':cf.config._asdict()
                    } for name,cf in part.corrfuncs.items()}

    for p in pickle.values():
        p['config'].pop('bins')

    dump( pickle,data_src+'corrfuncs')