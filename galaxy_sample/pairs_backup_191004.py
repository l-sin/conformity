import numpy as np
import two_point_tools as tp
import itertools
from itertools import chain, repeat
import matplotlib.pyplot as plt
from collections import namedtuple
from astropy.stats import bootstrap
    
class pair_list:
    def __init__(self,pairs,separations=None):
        
        if separations is None:
            print('Warning: pair_list declared without defining separation')
        elif len(pairs)!=len(separations):
            raise Exception("pairs and separations must have the same length, or be None")
        
        self.pairs = pairs
        self.count = len(self.pairs)
        self.weights = None
        self.separations = separations
        
        if separations is not None:
            self.sort(by='first')
        
        
    @classmethod
    def from_grid(cls,grid,regular=True,dtype='int64'):
        #opted for the most memory-efficient implementation
        count = sum(1 for i in grid.iter_pairs())
        pairs = np.empty((count,2)).astype(np.dtype(dtype))

        for i,p in enumerate(grid.iter_pairs()):
            pairs[i]=p
        
        if regular:
            separations = np.sqrt(np.sum( (grid.points[pairs[:,0]]-grid.points[pairs[:,1]])**2,
                                             axis=1 ))

            sel = np.logical_and(separations>0,separations<grid.spacing)
            pairs,separations = pairs[sel], separations[sel]
        else:
            separations = None
        
        return cls(pairs,separations)
    
    def select(self,selection,seltype=None):
        """
        selection can be of type galaxy_sample.samples.subsample,
        or (subsample,subsample),
        or boolean array with length==self.count
        """
        if seltype=='auto':
            
            subsample = selection
            mask = np.logical_and( subsample.acontains(self.first()), 
                                   subsample.acontains(self.second()) )
        elif seltype=='cross':
            if len(selection)!=2:
                raise Exception('selection should contain two subsamples in order to cross them.')
            
            sub1,sub2 = selection
            
            if any(np.logical_and(sub1.selection,sub2.selection)):
                # is there a better way to handle this?
                raise Exception('The subsamples to be crossed should not overlap.')
            
            
            
            mask = np.logical_or( 
                                np.logical_and( sub1.acontains(self.first()),
                                                sub2.acontains(self.second()) ),
                                np.logical_and( sub1.acontains(self.first()),
                                                sub2.acontains(self.second()) )
                                )
        else:
            if len(selection)!=self.count:
                raise Exception('Length of selection must equal length of pair list')
            mask = selection
            
        return pair_list( self.pairs[mask,:], self.separations[mask] )
    
    def first(self):
        return self.pairs[:,0]
    
    def second(self):
        return self.pairs[:,1]
    
    def downweight(self):
        # not compatible with subsampling at the moment  
        
        I = np.argsort(self.first())
        I_inv = np.argsort(np.arange(self.count+1)[I])
        
        self.sort(by='first')
        
        bins = np.arange( np.max( self.first() )+1 )
        
        counts = np.histogram( self.first(), bins=bins)[0]
        
        weights = np.array(list(chain.from_iterable(
                        list(repeat(c,c)) for c in counts
                        #repeat 1/c c-times, s.t. each primary has weight of 1
                  )))
        weights = weights[I_inv]
        
        return 1/weights
    
    def sort(self,by):
        case = {
                 'first':self.first(),
                 'second':self.second(),
                 'separation':self.separations
                }
        I = np.argsort(case[by])
        self.pairs = self.pairs[I,:]
        self.separations = self.separations[I]
    
    
class corrfunc:   
    
    def __init__(self, var, selection, weighted):
        Bins = namedtuple('bins',['scale','edges'])
        bins = Bins(scale='log',edges=np.arange(-1.5,1.1,0.25))
        
        Config = namedtuple('config',['var','selection','bins','weighted']) 
        self.config = Config(var, selection, bins, weighted)
        
    def match_target(self,target):
        """
        sample = galaxy_sample(...)
        e.g. selection = (cf for cf in sample.corrfuncs if cf.match_target(target))
        """
        return all( np.all(getattr(target,field) == getattr(self.config,field))
                    for field in target._fields)
    
    def match(self,var=None,selection=None,weighted=None):
        """
        sample = galaxy_sample(...)
        e.g. selection = (cf for cf in sample.corrfuncs if cf.match(...))
        
        Recommended method for selecting one or many corrfuncs from [galaxy sample].corrfuncs
        Intended to phase out the retarded 'match_target' method
        """
        fields = ['var','selection','weighted']
        target = dict()
        for field in fields:
            if eval(field) is not None:
                target[field]=eval(field)
        return all( np.all( target[field]==getattr(self.config,field)) for field in target )
    
    def compute(self):        
        var, selection, bins, weighted = self.config
        bin_scale, bin_edges = bins

        is_sub_of_parent = lambda sub: type(sub) is str and sub in self.parent.subsamples
        
        
        if is_sub_of_parent(selection):
            
            seltype = 'auto'
            selection = self.parent.subsamples[selection]
            
        elif (type(selection) is tuple 
              and len(selection)==2 
              and all(map(is_sub_of_parent, selection)) ):
            
            seltype = 'cross'
            selection = tuple(map( self.parent.subsamples.get, selection))

        elif len(selection) == self.parent.pair_list.count:
            
            seltype = None
            
        else:
            raise Exception('\'subsamples\' needs to be subsample of parent,'+
                            'or a tuple of (subsample,subsample),'+
                            'or a boolean array with the same length as pair_list')
            
        P = self.parent.pair_list.select(selection,seltype)

        
        if bin_scale == 'log':
            P.separations = np.log10(P.separations)
        mask = np.logical_and( bin_edges[0]<P.separations, P.separations<bin_edges[-1] )
        P = P.select(mask)
        P.sort(by='separation')
        
        f = self.parent.data[var]
        ff = f[P.first()]*f[P.second()]
        
        edges = self.config.bins.edges
        bin_counts = np.histogram(P.separations, edges)[0]
        ff = np.split(ff,np.cumsum(bin_counts)[:-1])
        rp = np.split(P.separations,np.cumsum(bin_counts)[:-1])
        
        if weighted:
            weights = P.downweight()
            weights = np.split(weights,np.cumsum(bin_counts)[:-1])
        else:
            weights = list(repeat(None,len(ff)))
        
        cf = []
        errorbars = []
            
        for pairs,wts in zip(ff,weights):

            if len(pairs)==0:
                cf.append(np.nan)
                errorbars.append(np.nan)
            else:
                cf.append( np.average(pairs,weights=wts) )
                
                if wts is None:
                    errorbars.append( np.std(bootstrap(pairs,bootfunc=np.average) ) )
                else:
                    errorbars.append( np.std(
                                        bootstrap(np.array( [pairs,wts] ).T,
                                                  bootfunc=lambda p: np.average(p[:,0],weights=p[:,1]))
                                            ) )
                
        self.results = cf 
        self.errorbars = errorbars
        
    def plot(self,ax,*args,**kwargs):
        bin_edges = self.config.bins.edges
        bins = bin_edges[:-1]+np.diff(bin_edges)/2
        
        if self.config.bins.scale=='log':
            bins = 10**bins
            
        #ax.plot(bins,self.results,*args,**kwargs)
        ax.errorbar(bins,self.results,self.errorbars,*args,**kwargs)
        
        return ax