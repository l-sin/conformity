import numpy as np
from . import comoving_distance

def projected_separation( angular_separation, redshift,
                          angular_units='degrees',output_units='comoving'):
    """
    converts angular separation on sky to projected separation
    
    assumes a flat cosmology.
    """
    if angular_units=='degrees':
        deg2rad = lambda deg: (deg/180.)*np.pi
        angular_separation = deg2rad( angular_separation )
    elif angular_units=='radians':
        pass
    else:
        raise Exception('angular_units must be either degrees or raidans.')
    
    z_grid = np.arange(min(redshift)-0.0002,max(redshift)+0.0002,0.0001)
    R_grid = comoving_distance(z_grid)
    R = np.interp(redshift,z_grid,R_grid)
        
    projected_separation = angular_separation*R
    
    if output_units=='comoving':
        return projected_separation
    elif output_units=='physical':
        return projected_separation/( 1+redshift )
    else:
        raise Exception('output_units must be either comoving or physical.')