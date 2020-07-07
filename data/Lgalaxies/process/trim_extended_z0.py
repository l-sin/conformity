import numpy as np
import h5py
from astropy.io import fits
import time

#the z=0 snapshot with full information (32GB) sits in Bruno's directory
#src = '/net/senna/scratch2/Henriques2015a/snaps/MR/'

#src = '/scratch/Documents/Conformity2019/data/Lgalaxies/'
fname = 'snap_MR_H15_extended_z0.00.fits'

with fits.open(src+fname,'readonly',memmap=True) as hdulist:
    hdu = hdulist[0]
    binary_table = hdulist[1]
    data = binary_table.data


fields = [
             'Type',
             'SnapNum',
             'CentralMvir',
             'CentralRvir',
             'DistanceToCentralGalX',
             'DistanceToCentralGalY',
             'DistanceToCentralGalZ',
             'PosX',
             'PosY',
             'PosZ',
             'VelX',
             'VelY',
             'VelZ',
             'Mvir',
             'Rvir',
             'Vvir',
             'Vmax',
             'InfallVmax',
             'InfallVmaxPeak',
             'InfallSnap',
             'InfallHotGas',
             'HotRadius',
             'OriMergTime',
             'MergTime',
             'EjectedMass',
             'HotGas',
             'ColdGas',
             'StellarMass',
             'BulgeMass',
             'DiskMass',
             'BlackHoleMass',
             'CoolingRadius',
             'CoolingRate',
             'CoolingRate_beforeAGN',
             'QuasarAccretionRate',
             'RadioAccretionRate',
             'Sfr'
]

t = time.time()
galaxies = {field: data[field] for field in fields}
print(time.time()-t)

#it takes an hour to read from the fits file

with h5py.File('/scratch/Documents/Conformity2019/data/Lgalaxies/processed/'+
               'snap_MR_H15_trimmed_z0.h5', 'w') as hf:
    g = hf.create_group('sample')
    for name in fields:
        g.create_dataset(name, data=galaxies[name], dtype=galaxies[name].dtype)
