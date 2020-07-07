import numpy as np
import two_point_tools as tp
import itertools
from itertools import chain, repeat
import matplotlib.pyplot as plt
from collections import namedtuple
from astropy.stats import bootstrap
import cosmic_distances as cosmo
    
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
    
    def eval_selection(self,sample,pairs):
        """
        sample: String, name of subsample of parent sample, or a 2-tuple of names
        pairs: String 'samehalo' or 'diffhalo', or boolean array of length equal to pair list
        """
        def resolve_sample(sample):
            if type(sample)==str:
                sample = self.parent.subsamples[sample]
            elif 'galaxy_sample.samples.subsample' in str(type(sample)):
                #what's the correct way to do this?
                pass
            else:
                raise Exception('sample could not be resolved')

            return sample
        
        
        try:
            sample=resolve_sample(sample)
            sample_mask = np.logical_and( sample.acontains(self.first()), 
                                          sample.acontains(self.second()) )

        except:
            sample1,sample2 = map(resolve_sample,sample)

            if any(np.logical_and(sample1.selection,sample2.selection)):
                raise Exception('The subsamples to be crossed should not overlap.')

            sample_mask = np.logical_or( 
                                        np.logical_and( sample1.acontains(self.first()),
                                                        sample2.acontains(self.second()) ),
                                        np.logical_and( sample2.acontains(self.first()),
                                                        sample1.acontains(self.second()) )
                                        )

        if pairs=='all':
            pairs_mask = np.ones(self.count,).astype('bool')
            
        elif pairs=='samehalo':
            group_ids = self.parent.groupcat.group_ids
            pairs_mask = group_ids[self.first()]==group_ids[self.second()]

        elif pairs=='diffhalo':
            group_ids = self.parent.groupcat.group_ids
            pairs_mask = group_ids[self.first()]!=group_ids[self.second()]

        else:
            if len(pairs)!=self.count:
                raise Exception('Length of selection must equal length of pair list')
            elif pairs.dtype != 'bool':
                raise Exception('selection must be boolean')

            pairs_mask = pairs


        selection = np.logical_and(sample_mask,pairs_mask)

        return selection

    def select(self,selection):
        
        selection = pair_list( self.pairs[selection,:], self.separations[selection] )
        
        if hasattr(self,'parent'):
            selection.parent = self.parent
        return selection
    
    def compute_rp(self,az='azimuthal_angle',pol='polar_angle',redshift='redshift',
                    angular_units='radians',output_units='physical'):

        az,pol,redshift = map(self.parent.data.get,(az,pol,redshift))

        if angular_units=='radians':
            pass
        elif angular_units=='degrees':
            az,pol = (az/180)*np.pi, (pol/180)*np.pi, 
            #convert to radians
        else:
            raise Exception('input angles must be in either radians or degrees')

        def sphere2cart(az,pol):
            x = np.cos(az)*np.sin(pol)
            y = np.sin(az)*np.sin(pol)
            z = -np.cos(pol)
            return x,y,z

        az1,az2 = az[self.pairs].T
        pol1,pol2 = pol[self.pairs].T

        xa,ya,za=sphere2cart(az1,pol1)
        xb,yb,zb=sphere2cart(az2,pol2)

        angular_separation = np.arccos(xa*xb + ya*yb + za*zb)
        mean_redshift = np.mean(redshift[self.pairs],axis=1)

        rp = cosmo.projected_separation( angular_separation, mean_redshift,
                                         angular_units=angular_units,
                                         output_units=output_units )

        return rp
    
    def first(self):
        return self.pairs[:,0]
    
    def second(self):
        return self.pairs[:,1]
    
    def downweight(self):
        # not compatible with subsampling at the moment  
        
        First = self.first()
        
        I = np.argsort(First)
        I_inv = np.argsort(np.arange(self.count+1)[I])
        
        
        bins = np.arange( np.max( First )+1 )
        
        counts = np.histogram( First, bins=bins)[0]
        
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
        
        
    def get(self,field):
        return self.parent.data[field][self.pairs]
    
    
class corrfunc:   
    
    def __init__(self, var, sample='full', pairs='all', weighted=True):
        Bins = namedtuple('bins',['scale','edges'])
        bins = Bins(scale='log',edges=np.arange(-1.5,1.1,0.25))
        
        Config = namedtuple('config',['var','sample','pairs','weighted','bins']) 
        self.config = Config( var, sample, pairs, weighted, bins )
        #
    
    
    def compute(self):        
        var, sample, pairs, weighted, bins = self.config
        bin_scale, bin_edges = bins
        
        self.selection = self.parent.pair_list.eval_selection(sample,pairs)
        P = self.parent.pair_list.select(self.selection)
        
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
    
    
    def match(self,var=None,sample='full',pairs='all',weighted=True):
        matches = []
        if var is not None:
            matches.append( self.config.var == var )

        matches.append( all(self.parent.pair_list.eval_selection(sample,pairs) == self.selection) )

        matches.append( self.config.weighted == weighted )

        return all( matches )
    
    
    def plot(self,ax,*args,**kwargs):
        bin_edges = self.config.bins.edges
        bins = bin_edges[:-1]+np.diff(bin_edges)/2
        
        if self.config.bins.scale=='log':
            bins = 10**bins
            
        #ax.plot(bins,self.results,*args,**kwargs)
        ax.errorbar(bins,self.results,self.errorbars,*args,**kwargs)
        
        return ax