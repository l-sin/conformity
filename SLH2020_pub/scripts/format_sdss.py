import numpy as np
import csv
from joblib import dump, load

src = '../../data/SDSS/processed/'

sdss_header=[]
with open(src+'vagc_mpa_yang_processed_header.txt','r') as f:
    for line in csv.reader(f):
        sdss_header.append(*line) 

sdss_data = []
with open(src+'vagc_mpa_yang_processed.txt','r') as f:
    for line in csv.reader(f):
        sdss_data.append(list(map(float,line)))
        
sdss_data = {field:np.array(data) for field,data in zip(sdss_header, zip(*sdss_data))}

sdss_data['azimuthal_angle'] = np.deg2rad(sdss_data['ra'])
sdss_data['polar_angle'] = (np.pi/2)-np.deg2rad(sdss_data['dec'])
sdss_data['StellarMass'] = sdss_data['mstar']
sdss_data['redshift'] = sdss_data['z']
sdss_data['Sfr'] = sdss_data['sfr']
sdss_data['fq_md_old']=sdss_data['fq_md']
sdss_data.pop('q')
sdss_data.pop('fq_md')

dst = '../data/SDSS/'
dump( sdss_data, dst+'data')