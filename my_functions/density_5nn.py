import numpy as np
import two_point_tools as tp
from itertools import chain
import cosmic_distances as cosmo

import sys
sys.path.append('/scratch/Documents/Conformity2019/')
from galaxy_sample import pair_list

def density_5nn( sample, spacing=(4*0.116,500),
                 az='azimuthal_angle',pol='polar_angle',redshift='redshift',
                 mass='StellarMass',tracerlim=10.4):
    
    c = 299792.458
    
    az,pol,redshift,mass = map(sample.data.get,(az,pol,redshift,mass))

    g = tp.spherical_grid( points = np.array([pol,az,c*redshift]).T,
                           spacing = spacing )
    rp5 = np.zeros(sample.count,)
    
    _,_,radial = zip(*g.points)
    radial = np.array(radial)
    separation_limit = cosmo.projected_separation(angular_separation = np.array([ g.angular_spacing ]),
                                          redshift = np.array([ min(redshift) ]),
                                          angular_units = 'radians',
                                          output_units = 'physical')[0]

    for pair_chunk in g.iter_pairs(chunk=True):
        pair_chunk = pair_list( np.array(pair_chunk), parent=sample )

        pair_chunk.separations = pair_chunk.compute_rp(
                                                         az='azimuthal_angle',
                                                         pol='polar_angle',
                                                         redshift='redshift',
                                                         angular_units='radians',
                                                         output_units='physical'
                                                        )
        cdz = np.abs(np.diff(radial[pair_chunk.pairs],axis=1)).reshape(-1)
        pair_chunk = pair_chunk.select( np.all([ 
                                                 mass[pair_chunk.second()]>tracerlim,
                                                 pair_chunk.first()!=pair_chunk.second(),
                                                 pair_chunk.separations<separation_limit, 
                                                 cdz<g.radial_spacing ], axis=0) )

        for gal in np.unique(pair_chunk.first()):
            nn = 4 if mass[gal]>tracerlim else 5
            try:
                rp5[gal] = sorted(pair_chunk.separations[ pair_chunk.first()==gal ])[nn-1]
            except:
                rp5[gal] = -1

    assert sum(rp5==0)==0
    if any(rp5==-1):
        print('Replacing rp5\'s larger than spacing with max')
        rp5[rp5==-1]=separation_limit
        
    l = cosmo.comoving_distance(redshift+(500/c)) - cosmo.comoving_distance(redshift-(500/c))
    
    return np.log10(5/(np.pi*rp5**2*l))