import numpy as np

def sphere2cart(az,pol):
    """
    converts azimuthal angle and polar angle (radians) on a unit sphere into cartesian coordinates
    """
    
    if any( np.logical_or( az<0,az>2*np.pi ) ) or any( np.logical_or( pol<0,pol>np.pi ) ):
        raise Exception('input angles must be in radians with the appropriate bounds')
    
    x = np.cos(az)*np.sin(pol)
    y = np.sin(az)*np.sin(pol)
    z = -np.cos(pol)
    
    return x,y,z