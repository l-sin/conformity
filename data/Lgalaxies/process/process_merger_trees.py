#!env/bin python
# coding: utf-8

import shutil
from astropy.io import fits
import time
import numpy as np
import h5py
import os


fnames = ['0_29','30_59','60_89','90_119','120_149','150_179','180_209',
          '210_239','240_269','270_299','300_329','330_359','360_389',
          '390_419','420_449','450_479','480_511']


for fname in fnames:
    t = time.time()
    src = '/export/data1/sinp/Downloads/SAM_trees/GalTree_MR_H15_'+fname+'.fits'
    hdulist = fits.open(src)
    hdu = hdulist[0]
    binary_table = hdulist[1]
    merger_trees = binary_table.data


    # Determine progenitors
    GalID_index_table = {v:k for k,v in enumerate(merger_trees['GalID'])}

    redshift_zero = merger_trees['SnapNum']==58
    progen_chains = [ np.where(redshift_zero)[0] ]

    # chains will contain, for z=0 galaxies and their most massive progenitors,
    # their INDICES in the table, and not the general galid

    for snapnum in range(1,58):    
        first_progen_id = merger_trees['FirstProgGal'][progen_chains[snapnum-1]]

        no_progen = first_progen_id==-1
        first_progen_id[no_progen] = merger_trees['GalID'][progen_chains[snapnum-1]][no_progen]

        first_progen_index = list(map( GalID_index_table.__getitem__, first_progen_id ))

        progen_chains.append(np.array(first_progen_index))


    chain_check = []

    for snapnum in range(1,57):  
        chain_check.append(all(np.logical_xor(
            merger_trees['FirstProgGal'][progen_chains[snapnum]]==merger_trees['GalID'][progen_chains[snapnum+1]],
                            progen_chains[snapnum] == progen_chains[snapnum+1]
                                                )))
    leaf_check = list(map(all,
                          merger_trees['MainLeafId'][progen_chains]==merger_trees['GalID'][progen_chains[-1]]))

    root_check = list(map(all,
                          merger_trees['TreeRootId'][progen_chains]==merger_trees['TreeRootId'][progen_chains[0]]))

    check = all(chain_check) and all(leaf_check) and all(root_check)

    if check:
        print('Chains check out')
    else:
        print('The chains are wrong')


    progen_chains = np.flipud(progen_chains)

    # Determine snapshot at which progenitor was only 50% of maximum mvir

    mmax = np.log10(np.amax(merger_trees['Mvir'][progen_chains],axis=0))
    snapform  = -np.ones(np.shape(mmax))
    prog_ind  = -np.ones(np.shape(mmax)) #index of half-mass progenitor in chain

    massfrac_prev = np.log10(merger_trees['Mvir'][progen_chains[0]]) - mmax

    for chain in progen_chains[1:]:

        massfrac = np.log10(merger_trees['Mvir'][chain]) - mmax
        half_mass_crossing = np.logical_and( massfrac>=np.log10(0.5), massfrac_prev<np.log10(0.5) )
        massfrac_prev = massfrac

        snapform[half_mass_crossing] = merger_trees['SnapNum'][chain[half_mass_crossing]]
        prog_ind[half_mass_crossing] = chain[half_mass_crossing]


    snapform=np.array(list(map(int,snapform)))
    prog_ind=np.array(list(map(int,prog_ind)))


    mock_sample = merger_trees[redshift_zero]

    with h5py.File('/scratch/Documents/Conformity2019/data/Lgalaxies/processed/z0_with_snapform_'+fname+'.h5', 'w') as hf:
        g = hf.create_group('sample')

        for name in mock_sample.names:
            g.create_dataset(name, data=mock_sample[name], dtype=mock_sample[name].dtype)

        g.create_dataset('snapform', data=snapform, dtype=snapform.dtype)
        
    print(time.time()-t)