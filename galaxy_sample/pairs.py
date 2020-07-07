import numpy as np
import two_point_tools as tp
import itertools
from itertools import chain, repeat
import matplotlib.pyplot as plt
from collections import namedtuple
from astropy.stats import bootstrap
import cosmic_distances as cosmo
    
class pair_list:
    def __init__(self,pairs,separations=None,parent=None):
        
        self.parent = parent
        
        if separations is not None and len(pairs)!=len(separations):
            raise Exception("pairs and separations must have the same length, or be None")
        
        self.pairs = pairs
        self.count = len(self.pairs)
        self.weights = None
        self.separations = separations
        
        if separations is not None:
            self.sort(by='first')
        
    @classmethod
    def from_grid(cls,grid,geometry,dtype='int64',parent=None):
        
        count = sum(1 for i in grid.iter_pairs())
        #pairs = np.empty((count,2)).astype(np.dtype(dtype))
        # if pair array with size of count is too large, raise Exception/Warning
        if geometry=='angular':
            _,_,radial = zip(*grid.points)
            radial = np.array(radial)
            separation_limit = cosmo.projected_separation(angular_separation = np.array([ grid.angular_spacing ]),
                                                          redshift = np.array([ min(parent.data['redshift']) ]),
                                                          angular_units = 'radians',
                                                          output_units = 'physical')[0]
        
        P,S = [],[]
        for chunk in grid.iter_pairs(chunk=True):
            
            pair_chunk = cls( np.array(chunk), parent=parent )
            
            if geometry=='regular':
                #TO BE CHECKED
                pair_chunk.separations = np.linalg.norm(grid.points[pair_chunk.first()]-grid.points[pair_chunk.second()],axis=1)
                pair_chunk = pair_chunk.select( np.logical_and(pair_chunk.separations>0,
                                                               pair_chunk.separations<=grid.spacing) )
                
            elif geometry=='angular':
                pair_chunk.separations = pair_chunk.compute_rp(
                                                                 az='azimuthal_angle',
                                                                 pol='polar_angle',
                                                                 redshift='redshift',
                                                                 angular_units='radians',
                                                                 output_units='physical'
                                                                )
                
                cdz = np.abs(np.diff(radial[pair_chunk.pairs],axis=1)).reshape(-1)

                pair_chunk = pair_chunk.select( np.all([ pair_chunk.first()!=pair_chunk.second(),
                                                         pair_chunk.separations<separation_limit, cdz<grid.radial_spacing ], axis=0) )

            else:
                separations = None
                
            P.append(pair_chunk.pairs)
            S.append(pair_chunk.separations)
            
        P = np.array(list(chain(*P)))
        S = np.array(list(chain(*S)))
        pairs = cls(P,separations=S,parent=parent)
        
        return pairs
    
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
        if type(pairs)==str:
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
    #remove match() in this version. Initialization and duplicate control now completely on user-side
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
        stdev = []
            
        for pairs,wts in zip(ff,weights):

            if len(pairs)==0:
                cf.append(np.nan)
                errorbars.append(np.nan)
                stdev.append(np.nan)
            else:
                cf.append( np.average(pairs,weights=wts) )
                
                if wts is None:
                    errorbars.append( np.std(bootstrap(pairs,bootfunc=np.average) ) )
                    stdev.append( np.std(pairs) )
                else:
                    errorbars.append( np.std(
                                        bootstrap(np.array( [pairs,wts] ).T,
                                                  bootfunc=lambda p: np.average(p[:,0],weights=p[:,1]))
                                            ) )
                    wmean = np.average( pairs,weights=wts )
                    stdev.append( np.sqrt(np.average((pairs-wmean)**2, weights=wts)) )
        
        self.results = cf 
        self.errorbars = errorbars
        self.stdev = stdev

    
    def plot(self,ax,mode=('errorbar','errorbars'),*args,**kwargs):
        """
        mode: (plotmode,errormode)
        plotmode: 'plot','errorbar', or 'fill_between'
        errormode: 'errorbars' or 'stdev', ignored if plotmode is 'plot'
        """
        
        bin_edges = self.config.bins.edges
        bins = bin_edges[:-1]+np.diff(bin_edges)/2
        
        if self.config.bins.scale=='log':
            bins = 10**bins
            ax.set_xscale('log')
            ax.set_xlim(10**bin_edges[[0,-1]])
            
        plotmode,errormode=mode
        
        if plotmode=='plot':
            ax.plot(bins,self.results,*args,**kwargs)
        elif plotmode=='errorbar':
            ax.errorbar(bins,self.results,self.__getattribute__(errormode),*args,**kwargs)
        elif plotmode=='fill_between':
            ax.fill_between(bins,
                            np.array(self.results)-np.array(self.__getattribute__(errormode)),
                            np.array(self.results)+np.array(self.__getattribute__(errormode)),
                            *args,**kwargs)
        
        return ax