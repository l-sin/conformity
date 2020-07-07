import numpy as np
import two_point_tools as tp
import itertools
from itertools import chain, repeat
from functools import reduce
from .pairs import pair_list,corrfunc

class galaxy_sample:
    
    def __init__(self,galaxies_dict):
        self.data = dict()
        
        lens = set(map(len,galaxies_dict.values()))
        if len(lens)!=1:
            raise Exception("galaxies_dict must consist of entries with equal length")
        self.count, = lens # should be fine?
        
        for field,data in galaxies_dict.items():
            self.add_field(field,data)
        
        self.subsamples=dict()
        self.define_subsample(name='full', 
                              selection=np.ones(self.count,).astype('bool'))
        
        self.corrfuncs=[]
    
    def __repr__(self):
        return 'galaxy_sample with {} galaxies, and {} fields.'.format(self.count,len(self.data))
    
    
    def add_field(self,field,data):
        if field in self.data:
            raise Exception("field already exists in data, use update_field instead")
        if len(data)!=self.count:
            raise Exception("The dimension of data must match the size of the galaxy sample")
        
        self.data[field] = data
        
        
    def update_field(self,field,data):
        if field not in self.data:
            raise Exception("field must exist in order to be updated, use add_field instead")
        if len(data)!=self.count:
            raise Exception("The dimension of data must match the size of the galaxy sample")
        
        self.data[field] = data
        
    
    def cut_sample(self,cut):
        for field in self.data:
            self.data[field] = self.data[field][cut]
        self.count = np.sum(cut)
        
        for subsample in self.subsamples.values():
            subsample.count = np.sum(np.logical_and(cut,subsample.selection))
            subsample.selection = subsample.selection[cut]
            
        if hasattr(self,'pair_list'):
            print('cut_sample invalidates indexing in pair_list,'+
                  'deleting pair_list and clearing corrfuncs.')
            del self.pair_list
            self.corrfuncs = []
    
    def define_subsample(self, name, selection):
        self.subsamples[name] = subsample(self,selection)
        
    def define_pairs(self, pair_list):
        self.pair_list = pair_list
        self.pair_list.parent = self
        
    def define_corrfunc(self, var, selection='full', weighted=True):

        if not hasattr(self,'pair_list'):
            raise Exception('pair_list is required to compute corrfunc, but is not yet defined.')

        if any( cf.match(var,selection,weighted) for cf in self.corrfuncs ):
            print('A corr. func. with the same config. already exists in corrfuncs. Passing.')
            pass
        else:
            cf = corrfunc(var, selection, weighted)
            cf.parent = self
            cf.compute()
            self.corrfuncs.append(cf)
        
        
    def split_by_cleanliness(self,varlist,subsample=None):

        assert type(varlist)==list, "Argument varlist must be a list of variable name(s)."

        if subsample is None:
            subsample = np.ones(self.count,)
        
        var_values = map(self.data.__getitem__, varlist)
        
        clean = np.all( 
                        [
                         subsample,
                         *[ li!=-np.inf for li in var_values ],
                         *[ np.isnan(li)==False for li in var_values ]
                        ],
                        axis=0
                        )

        unclean = np.all( [subsample, clean==False], axis=0 )

        return clean, unclean

    
    def get(self,fields,order=None):
        if order is not None:
            I = np.argsort(self.data[order])
            return {field:self.data[field][I] for field in fields}
        else:
            return {field:self.data[field] for field in fields}


class subsample:
    def __init__(self,parent,selection):
        self.parent = parent
        self.selection = selection
        self.count = np.sum(selection)

    def __getitem__(self,key):
        return self.parent.data[key][self.selection]

    def __contains__(self,index):
        return self.selection[index]

    def acontains(self,indices):
        return np.array(list(map( self.__contains__, indices )))