import numpy as np
import two_point_statistics as tp
import my_functions as mf

def smooth2d(x,y,z,func,smoothness,smoothBy="length",normalize=False):
    
    f = np.zeros(np.shape(z))

    if normalize:
        normed = lambda a: np.copy(a)/np.std(a)
        x,y = normed(x),normed(y)
    
    if smoothBy=='length':
        xbins = np.arange(min(x)-smoothness, max(x)+smoothness,smoothness)
        ybins = np.arange(min(y)-smoothness, max(y)+smoothness,smoothness)
    elif smoothBy=='nearest_k':
        error('Not yet built')

    x_hash = mf.bin_hash(x,xbins).astype(int)
    y_hash = mf.bin_hash(y,ybins).astype(int)

    C = [ [[]for j in range(max(y_hash)+1)]    for i in range(max(x_hash)+1)]

    for i in range(len(x)):
        # for every galaxy, put its index in its corresponding bin in C
        C[x_hash[i]][y_hash[i]].extend([np.int32(i)])

    ilim, jlim = np.shape(C)

    for i in range(max(x_hash)):
        for j in range(max(y_hash)):
            target = tp.make_target_2d(C,i,j,ilim-1,jlim-1)
            P = tp.all_pairs(C[i][j], target)

            num_neighs=np.histogram( [p[0] for p in P], np.arange(0,len(x)+1,1) )[0]
            counter=0
            for gal in C[i][j]:
                neighs = [p[1] for p in P[counter:counter+num_neighs[gal]]]
                #print(num_neighs[gal],len(neighs))
                f[gal] = func(z[neighs])
                counter+=num_neighs[gal]
                
    return f