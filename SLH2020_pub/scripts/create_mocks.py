import numpy as np
import h5py
import sys
sys.path.insert(0,'/scratch/Documents/Conformity2019/')
import cosmic_distances as cosmo
import time
from joblib import dump, load

sys.path.insert(0,'/scratch/Documents/Conformity2019/data/SDSS/')
from emulate_sdss_completeness import *

c,h = 299792.458, 0.677

# Read the full volume with all relevant fields
with h5py.File('/scratch/Documents/Conformity2019/data/Lgalaxies/processed/'+
               'snap_MR_H15_trimmed_z0.h5', 'r') as hf:
    galaxies = { k: v.__array__() for k,v in hf['sample'].items()}

file_numbers = ['0_29','30_59','60_89','90_119','120_149','150_179','180_209','210_239','240_269','270_299',
                '300_329','330_359','360_389','390_419','420_449','450_479','480_511']

fields = ['FOFCentralGal', 'GalID', 'PosX', 'PosY', 'PosZ', 'StellarMass']

src = '/scratch/Documents/Conformity2019/data/Lgalaxies/processed/'
fstring = 'z0_with_snapform_{}.h5'

#read
Addition = {field:[] for field in fields}

for field in fields:
    for fid in file_numbers:
        with h5py.File(src+fstring.format(fid),'r') as hf:
            Addition[field].extend( list(hf['sample'][field].__array__()) )

    Addition[field] = np.array(Addition[field])

# Matching datasets
StellarMassCut = np.log10(galaxies['StellarMass']) + 10 - np.log10(h) > 9.0
for field in galaxies:
    galaxies[field] = galaxies[field][StellarMassCut]

StellarMassCut = np.log10(Addition['StellarMass']) + 10 - np.log10(h) > 9.0
for field in Addition:
    Addition[field] = Addition[field][StellarMassCut]

IdxTable = {v:k for k,v in enumerate(zip(Addition['StellarMass'],Addition['PosX']))}

match = -np.ones(len( galaxies['Type'] ),).astype('int')
for i,(mass,x) in enumerate(zip(galaxies['StellarMass'], galaxies['PosX'])):
    match[i] = IdxTable[(mass,x)]

if not all(galaxies['PosZ']==Addition['PosZ'][match]):
    raise Exception('Failed to match the two catalogues')

galaxies['FOFCentralGal'] = Addition['FOFCentralGal'][match]
galaxies['GalID'] = Addition['GalID'][match]



#Unit conversion
mass_fields = ['CentralMvir','Mvir','InfallHotGas','EjectedMass','HotGas','ColdGas',
               'StellarMass','BulgeMass','DiskMass','BlackHoleMass']

distance_fields = ['CentralRvir','DistanceToCentralGalX','DistanceToCentralGalY','DistanceToCentralGalZ',
                   'PosX','PosY','PosZ','Rvir','Vvir','Vmax','InfallVmax','InfallVmaxPeak','HotRadius',
                   'CoolingRadius']

for field in mass_fields:
    galaxies[field] = np.log10(galaxies[field]) + 10 - np.log10(h)

for field in distance_fields:
    galaxies[field] /= h
    
galaxies['Sfr'] = np.log10(galaxies['Sfr'])



# Mock observe
def cart2sphere(x,y,z):
    r = np.sqrt( x**2 + y**2 + z**2 )
    az = np.arctan2(y,x)
    pol = np.arccos(z/r)
    return r,az,pol

r, az, pol = cart2sphere(galaxies['PosX']-714/2,galaxies['PosY']-714/2,galaxies['PosZ']-714/2)

v = np.array([
                galaxies['VelX'],
                galaxies['VelY'],
                galaxies['VelZ']
                ]).T

r_unit = np.array([
                    np.cos(az)*np.sin(pol),
                    np.sin(az)*np.sin(pol),
                    np.cos(pol)
                    ]).T

v_r = np.sum(v*r_unit,axis=1)

z_grid = np.arange(0,0.1,0.0001)
D_grid = cosmo.comoving_distance(z_grid)
z = np.interp(r,D_grid,z_grid)
z += v_r/c

galaxies['redshift']=z
galaxies['azimuthal_angle']=az+np.pi
galaxies['polar_angle']=pol

mass_completeness = emulate_sdss_completeness( galaxies['redshift'] )
cut = np.all(  [
                0.02<galaxies['redshift'],galaxies['redshift']<0.08,
                galaxies['StellarMass']>9.0,
                galaxies['StellarMass']>mass_completeness
                ],axis=0  )

for field in galaxies:
    galaxies[field] = galaxies[field][cut]
    
# Divide into subvols
X_split = galaxies['PosX']>357
Y_split = galaxies['PosY']>357
Z_split = galaxies['PosZ']>357

Selections = [ np.all([X_split, Y_split, Z_split],axis=0),
               np.all([X_split, Y_split, Z_split==False],axis=0),
               np.all([X_split, Y_split==False, Z_split],axis=0),
               np.all([X_split, Y_split==False, Z_split==False],axis=0),
               np.all([X_split==False, Y_split, Z_split],axis=0),
               np.all([X_split==False, Y_split, Z_split==False],axis=0),
               np.all([X_split==False, Y_split==False, Z_split],axis=0),
               np.all([X_split==False, Y_split==False, Z_split==False],axis=0) ]

dst = '../data/LGalaxies/'
dump(galaxies,dst+'full_volume_mock_data')

for i,sel in enumerate(Selections):
    part = {k:v[sel] for k,v in galaxies.items()}
    dump( part, dst+'{}/'.format(i)+'data' )