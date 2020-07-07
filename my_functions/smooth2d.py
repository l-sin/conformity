import numpy as np
import two_point_tools as tp
import my_functions as mf
from galaxy_sample import pair_list

def smooth2d(x,y,z,func,smoothness,normalize=False):
    
    f = np.empty(z.shape)
    f.fill(np.nan)

    if normalize:
        normed = lambda a: np.copy(a)/np.std(a)
        x,y = normed(x),normed(y)
    
    #if smoothBy=='length':
    #    xbins = np.arange(min(x)-smoothness, max(x)+smoothness,smoothness)
    #    ybins = np.arange(min(y)-smoothness, max(y)+smoothness,smoothness)

    g = tp.grid( list(zip(x,y)), spacing=smoothness )
    
    for pair_chunk in g.iter_pairs(chunk=True):
        pair_chunk = pair_list( np.array(pair_chunk) )

        pair_chunk.separations = np.linalg.norm(g.points[pair_chunk.first()]-g.points[pair_chunk.second()],axis=1)
        
        pair_chunk = pair_chunk.select( np.logical_and(pair_chunk.first()!=pair_chunk.second(),
                                                       pair_chunk.separations<smoothness) )
        
        
        pair_chunk.sort(by='first')
        try:
            bin_counts = np.histogram(pair_chunk.first(),
                                  bins=np.arange( np.min(pair_chunk.first()),np.max(pair_chunk.first())+2,1 )
                                  )[0]
        except:
            if pair_chunk.count==0:
                continue
            else:
                raise
            
        #print(np.split(pair_chunk.pairs,np.cumsum(bin_counts)[:-1]))
        
        for sub_chunk in np.split(pair_chunk.pairs,np.cumsum(bin_counts)[:-1]):
            if sub_chunk.shape[0]!=0:
                assert len(set(sub_chunk[:,0]))==1
                f[sub_chunk[0,0]]=func(z[sub_chunk[:,1]])
                
    IsNan = np.isnan(f)
    if any(IsNan):
        print('smooth2d returned {} values without estimates, imputing with their own value'.format(np.sum(IsNan)))
        f[IsNan] = z[IsNan]
                
    return f