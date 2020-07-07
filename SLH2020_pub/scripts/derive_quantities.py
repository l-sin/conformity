import numpy as np
import h5py
import sys
sys.path.insert(0,'/scratch/Documents/Conformity2019/')
import my_functions as mf
import two_point_tools as tp
from galaxy_sample import galaxy_sample,pair_list,halo_mass_model,tinker_group_finder
import time
from joblib import dump, load

c,h = 299792.458, 0.677
AM_redshift_bins = np.arange(0.02,0.09,0.01)
AM_src = '../data/LGalaxies/'


sample = sys.argv[1]
if sample=='sdss':
    data_sources = ['../data/SDSS/']
    
elif sample=='mock':
    data = load(AM_src+'full_volume_mock_data')
    full = galaxy_sample( data )

    for SHAM in (True,False):
        AM = halo_mass_model.RedshiftDependentAbundanceMatching(AM_redshift_bins, src=None, load_pop=False, SHAM=SHAM)
        AM.define_population(full)
        AM.dump_population(dst=AM_src)
    
    data_sources = ['../data/LGalaxies/{}/'.format(i) for i in range(8)]


for data_src in data_sources:
    data = load(data_src+'data')
    part = galaxy_sample( data )

    g = tp.spherical_grid( points = np.array([ part.data['polar_angle'],
                                               part.data['azimuthal_angle'],
                                               c*part.data['redshift'] ]).T,
                           spacing = (0.116,500) )

    part.define_pairs( pair_list.from_grid( g, geometry='angular', parent=part) )
    dump( part.pair_list.pairs, data_src+'pairs')
    dump( part.pair_list.separations, data_src+'separations')

    #q,fq,delta
    part.add_field( 'density', mf.density_5nn(part) )
    
    qline = lambda mass: -0.45*(mass - 10) - 10.85
    part.add_field( 'q', part.data['Sfr']-part.data['StellarMass']<qline(part.data['StellarMass']) )

    part.add_field( 'fq_md', mf.smooth2d(part.data['StellarMass'],part.data['density'],
                                         part.data['q'],
                                         np.mean,0.1,normalize=True))
    part.add_field( 'Delta_md', part.data['q']-part.data['fq_md'] )


    #group finding
    AM = halo_mass_model.RedshiftDependentAbundanceMatching(src=AM_src,SHAM=False,redshift_bins=AM_redshift_bins)

    rscaleMod,veldispMod,B = 1.2, 3.5, 40

    gf = tinker_group_finder(
                             part,
                             sky_fraction=None,
                             az='azimuthal_angle',
                             pol='polar_angle',
                             redshift='redshift',
                             stellarmass='StellarMass',
                             spacing=(0.04,2000),
                             load_pairs=True,
                             pairs_src=data_src
                             )
    gf.grid.neighbour_elems_precompute = {elem:gf.grid.neighbours(elem) for elem in gf.grid.keys()}

    iter_results = gf.run(AM, B, rscaleMod, veldispMod, max_iter=5, verbose=sample=='mock')
    groupIDs,_ = iter_results[-1]
    part.add_field('ObsGrNr',groupIDs)

    # Dump
    dump( part.data, data_src+'data' )