import numpy as np
from scipy import integrate


def comoving_distance(z):
    """
    converts redshift into comoving distance

    assumes flat cosmology, and that redshift is entirely due to Hubble flow
    """
    c  = 299792.458; # km/s
    H0 = 67.7;       # km/s /Mpc
    OM = 0.3;
    OL = 0.7;
    
    E = lambda z: (OM*(1+z)**3 + OL)**(-0.5)
    integral = lambda z : (c/H0)*integrate.quad(E,0,z)[0]

    D = np.array(list(map(integral,z)))

    return D