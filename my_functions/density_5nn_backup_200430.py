import numpy as np
import two_point_tools as tp
from itertools import chain
import cosmic_distances as cosmo

def density_5nn( sample, spacing=(4*0.116,500),
                 az='azimuthal_angle',pol='polar_angle',redshift='redshift',
                 mass='StellarMass',tracerlim=10.4):
    # spacing = (4*0.116,500) works well heuristically
    az,pol,redshift,mass = map(sample.data.get,(az,pol,redshift,mass))
    
    sin_pol,cos_pol = np.sin(pol),np.cos(pol)
     
    def quick_angular_separation(az1,sin_pol1,cos_pol1,az2,sin_pol2,cos_pol2):
        return np.arccos(cos_pol1*cos_pol2+sin_pol1*sin_pol2*np.cos(az1-az2))

    g = tp.spherical_grid( points = np.array([pol,az,3e5*redshift]).T,
                           spacing = spacing )
    rp5 = np.zeros(sample.count,)

    for elem in g:
        neighbouring_tracers = []
        num_tracer_neighs = [0]

        neighbours = g.neighbours(elem)
        target = np.array(list(chain( *[t for t in map(g.get, neighbours) if t is not None] )))
        target = target[mass[target]>tracerlim]

        tracer_pairs = np.array([ p for p in tp.all_pairs(g[elem],target) ])
        if len(tracer_pairs)>0:
            angular_separation = quick_angular_separation(
                                        az[tracer_pairs[:,0]],
                                        sin_pol[tracer_pairs[:,0]],
                                        cos_pol[tracer_pairs[:,0]],
                                        az[tracer_pairs[:,1]],
                                        sin_pol[tracer_pairs[:,1]],
                                        cos_pol[tracer_pairs[:,1]]
                                                ).reshape(len(tracer_pairs),)

            mean_redshift = np.mean(redshift[tracer_pairs],axis=1)

            rp = cosmo.projected_separation( angular_separation, mean_redshift,
                                             angular_units="radians", output_units='physical' )
            cdz = 3e5*np.abs(redshift[tracer_pairs[:,0]] - redshift[tracer_pairs[:,0]])
            sel = np.all([
                            tracer_pairs[:,0]!=tracer_pairs[:,1],
                            rp<40,
                            cdz<500
                         ],axis=0)
            tracer_pairs = tracer_pairs[sel]
            rp = rp[sel]

            num_tracer_neighs = np.histogram(tracer_pairs[:,0],bins=np.arange(sample.count+1) )[0][g[elem]]

        if min(num_tracer_neighs)<5:
            raise Exception('Gridding is too small to find 5th-NN, increase spacing[0]')

        for gal in g[elem]:
            nn = 4 if mass[gal]>tracerlim else 5
            rp5[gal] = sorted(rp[ tracer_pairs[:,0]==gal ])[nn-1]


    l = cosmo.comoving_distance(redshift+(500/3e5)) - cosmo.comoving_distance(redshift-(500/3e5))

    return np.log10(5/(np.pi*rp5**2*l))