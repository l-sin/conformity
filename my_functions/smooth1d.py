import numpy as np
import my_functions as mf

def smooth1d(x,y,xs,func):
    
    x,y = np.array(x),np.array(y)

    #Precomputation
    x_0 = np.copy(x)

    x,I = np.sort(x),np.argsort(x)
    y = y[I]
    xs = xs/2
    f = np.zeros(np.shape(x))    
    
    dum = np.arange(len(x))
    dum = dum[I]
    I_inv = np.argsort(dum)
    
    assert(all(x[I_inv]==x_0)) 

    x_min = x[0]
    x_max = x[-1]
    
    x_edges = np.arange(min(x)-xs, max(x)+xs, xs)
    hx = np.histogram(x,x_edges)[0]
    
    x_hash = np.array(mf.bin_hash(x, x_edges)).astype(int)
    
    left_ind = 0
    right_ind = 1 + hx[x_hash[0]] + hx[x_hash[0]+1]
    
    for i in range(len(x)):
        if x[i]-x[left_ind]>xs:
            while x[i]-x[left_ind]>xs:
                left_ind+=1
        elif x[i]-x[left_ind]<xs:
            while x[i]-x[left_ind]<xs and left_ind>0:
                left_ind-=1

        if x[right_ind]-x[i]>xs:
            while x[i]-x[right_ind]>xs:
                right_ind-=1
        elif x[right_ind]-x[i]<xs:
            while x[right_ind]-x[i]<xs and right_ind<len(x)-1:
                right_ind+=1            
        
        f[i] = func( y[left_ind:right_ind+1] )

    f = f[I_inv]
    
    return f