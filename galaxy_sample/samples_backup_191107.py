import numpy as np
import two_point_tools as tp
import itertools
from itertools import chain, repeat
from functools import reduce
from .pairs import pair_list,corrfunc
import my_functions as mf


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
        
    def define_groups(self, group_ids):
        self.groupcat = groupcat(self,group_ids)
        
    def define_corrfunc(self, *args, **kwargs):

        if not hasattr(self,'pair_list'):
            raise Exception('pair_list is required to compute corrfunc, but is not yet defined.')

        if any( cf.match(*args, **kwargs) for cf in self.corrfuncs ):
            print('A corr. func. with the same config. already exists in corrfuncs. Passing.')
            pass
        else:
            cf = corrfunc(*args, **kwargs)
            cf.parent = self
            cf.compute()
            self.corrfuncs.append(cf)
        
        
    def compute_fq(self,var,sub):
        if type(sub)==str:
            sub = self.subsamples[sub]
        elif 'galaxy_sample.samples.subsample' not in str(type(sub)):
            raise Exception('subsample must be name, or subsample')

        fq = -np.ones(sub.count,)

        sel = sub[var]==-np.inf
        if any(sel):
            print( 'input sample contains {} -inf values '.format(np.sum(sel))+
                   'out of {} galaxies,'.format(sub.count)+
                   'imputing their fq with their average q, {0:.5}'.format(np.mean(self.data['q'][sub.selection][sel])) )
        fq[sel] = np.mean(self.data['q'][sub.selection][sel])

        sel = sub[var]!=-np.inf 
        fq[sel] = mf.smooth1d(sub[var][sel].astype('float64'),sub['q'][sel],0.1,np.mean)

        return fq

    
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
    
    
class groupcat:
    def __init__(self,parent,group_ids):    
        self.parent = parent
        self.group_ids = group_ids
        
        self.groups=dict()
        for idx,Id in enumerate(group_ids):
            if Id in self.groups:
                self.groups[Id].append(idx)
            else:
                self.groups[Id]=[idx]
        
    @classmethod
    def from_dict(cls,parent,group_dict):
        self.parent = parent
        self.groups = group_dict
        
        self.group_ids = -np.ones(self.parent.count,)
        for Id,members in self.groups.items():
            self.group_ids[members] = Id
            
    def define_centrals(self,mask=None,crit='StellarMass'):
        if mask is None:
            mask = np.zeros(self.parent.count,).astype('bool')
            for group in self.groups.values():
                maxmass = np.argmax(self.parent.data[crit][group])
                mask[group[maxmass]] = True

        self.centrals = subsample(self.parent, mask)
        self.satellites = subsample(self.parent, np.logical_not(mask))