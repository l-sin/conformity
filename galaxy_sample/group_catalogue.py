import numpy as np
from collections import defaultdict

import sys
sys.path.append('/scratch/Documents/Conformity2019')
import my_functions as mf
from galaxy_sample import halo_mass_model

class group_catalogue:
    def __init__(self,sample,AM=None):

        TrueGroups = dict()
        for idx,GrNr in enumerate(sample.data['FOFCentralGal']):
            if GrNr not in TrueGroups:
                TrueGroups[GrNr] = [idx]
            else:
                TrueGroups[GrNr].append(idx)
                
        Groups = dict()
        for idx,GrNr in enumerate(sample.data['ObsGrNr']):
            if GrNr not in Groups:
                Groups[GrNr] = [idx]
            else:
                Groups[GrNr].append(idx)

        self.TrueGroupIDs = sample.data['FOFCentralGal']        
        self.TrueGroups = TrueGroups
        self.ObsGroupIDs = sample.data['ObsGrNr']
        self.ObsGroups = Groups
        
        
        self.ObsRich  = np.array(list(map(len,self.ObsGroups.values())))
        self.TrueRich = np.array(list(map(len,self.TrueGroups.values())))
        
        if AM is None:
            AM = halo_mass_model.AbundanceMatching(load_pop=False)
            
        self.TrueMass = AM.get_features(sample,self.TrueGroups,training=True)
        self.TrueMass = self.TrueMass[0] if len(self.TrueMass)==2 else self.TrueMass
        try:
            self.ObsMass = AM.predict(sample,self.ObsGroups)
        except:
            print('Unable to compute observed halo mass based on given halo mass model.')
        
    def IoU(self,ko,kt):
        o = set( self.ObsGroups[ko] )
        t = set( self.TrueGroups[kt] )
        return len(o.intersection(t)) / len(o.union(t))
    
    def best_match(self,base):
        #return the id of the groups accompanying catalogue with the highest IoU with itself
        if base=='true':
            Groups = self.TrueGroups
            CounterpartIDs = self.ObsGroupIDs
        elif base=='obs':
            Groups = self.ObsGroups
            CounterpartIDs = self.TrueGroupIDs
        else:
            raise ValueError()
            
        best_matches = []
        for k,v in Groups.items():
            counterparts = np.unique( CounterpartIDs[v] )
            foo = ( ((self.IoU(group,k),group) for group in counterparts) if base=='true' else
                    ((self.IoU(k,group),group) for group in counterparts) )
            best_matches.append( max( foo, key=lambda x: x[0]) )
            
        return best_matches
    
    def two_way_match_fraction(self,base,threshold,binning):
        
        if binning[0] == 'mass':
            binning[0]=self.ObsMass if base=='obs' else self.TrueMass
        elif binning[0] == 'richness':
            binning[0]=self.ObsRich if base=='obs' else self.TrueRich
        else:
            raise ValueError()
        
        IoUs,_ = zip(*self.best_match(base))
        has_two_way_match = np.array(IoUs)>threshold

        fractions = [np.mean(np.array(has_two_way_match)[members]) 
                     for _,members in mf.binmembers(*binning)]
        bins = np.mean( [binning[1][:-1],binning[1][1:]], axis=0 )

        return bins,fractions
    
    def group_stats(self,ko,kt):
        o = set( self.ObsGroups[ko] )
        t = set( self.TrueGroups[kt] )

        completeness = len(o.intersection(t))/len(t)
        contamination1 = len(o.difference(t))/len(t)
        contamination2 = len(o.difference(t))/len(o)

        return completeness,contamination1,contamination2
    
    def completeness_purity(self,base,threshold,binning): 
        Matches = [(ko,kt) for ko,iou,kt in zip(self.ObsGroups,*zip(*self.best_match('obs'))) 
                   if iou>threshold and len(self.ObsGroups[ko])>1 and len(self.TrueGroups[kt])>1]
        
        if base=='true':
            Richness = [ len(self.TrueGroups[kt]) for _,kt in Matches]
        elif base=='obs':
            Richness = [ len(self.ObsGroups[ko]) for ko,_ in Matches]
        else:
            raise ValueError()

        stats=[]
        for b,m in mf.binmembers(np.array(Richness),binning[1]):
            
            if m.size==0:
                c0,c1,c2 = (),(),()
            else:
                c0,c1,c2 = zip(*[self.group_stats(ko,kt) 
                                 for ko,kt in np.array(Matches)[m]])
            stats.append((c0,c1,c2))
        stats = list(zip(*stats))
            
        bins = list(zip(binning[1][:-1],binning[1][1:]))
        
        return bins,stats