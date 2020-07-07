import numpy as np

def binmembers(data,edges):
    """
    Given data vector and bin edges, generate corresponding bin centers and members of bin
    """
    
    mask = np.logical_and( edges[0]<data, data<edges[-1] )
    
    I = np.argsort(data)
    I_inv = np.arange(len(data))[I]
    data=data[I]
    mask=mask[I]
    
    bins = edges[:-1]+np.diff(edges)/2
    bin_counts = np.histogram(data, edges)[0]
    
    II = np.split(I_inv[mask],np.cumsum(bin_counts))
    
    for b,members in zip(bins,II):
        yield b,members