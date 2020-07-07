import numpy as np

def bin_hash(items,bin_edges,return_bins=False):
    
    items = np.array(items)
    bh = -np.ones(items.size)
    d = np.diff(bin_edges)[0]
    
    inrange = np.logical_and(items>=bin_edges[0], items<=bin_edges[-1])
    
    items = items[inrange]
    items -= bin_edges[0]
    h = (items - items%d)/d
    
    bh[inrange] = h
    bh[inrange==0] = np.nan
    
    if return_bins:
        bins = bin_edges[:-1]+np.diff(bin_edges)/2
        return bins,bh
    else:
        return bh