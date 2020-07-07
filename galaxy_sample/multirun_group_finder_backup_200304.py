import numpy as np
from itertools import chain
from statistics import mode, StatisticsError

import sys
sys.path.append('/scratch/Documents/Conformity2019/')
import my_functions as mf
import cosmic_distances as cosmo
import two_point_tools as tp
from galaxy_sample import pair_list,fof_group_finder,Group

#spacing = (0.116,2000)

class multirun_group_finder(fof_group_finder):
    
    def multirun(self,params):
        """
        params: tuple of param, with each param being (filter, l_p,l_z)
        filter: boolean function, with filter(group) returning True for groups to keep on that run

        pseudocode for multirun
        initialize 'already-grouped galaxies' with zeros
        for filter,param in params:
            run link_pairs with param
            knock out pairs which contain already-grouped galaxies
            form group from remaining pairs
            collect groups with filter(group)==True, and update already-grouped galaxies
        """
        multirun_groups = []
        grouped = np.zeros(self.sample.count).astype('bool')

        for f,l_p,l_z in params:
            P = self.link_pairs(l_p,l_z)
            contains_grouped = np.any(grouped[P.pairs],axis=1)
            P = P.select( np.logical_not(contains_grouped) )

            groupIDs,groups = self.merge_links(P)

            for group_members in f(self,groups):
                #grouped[ group.members ] = True
                grouped[ group_members ] = True
                multirun_groups.append( group_members )

        groupIDs = -np.ones(self.sample.count,).astype('int')
        multirun_groups = {GrNr:group for GrNr,group in enumerate(multirun_groups)}
        for GrNr,group in multirun_groups.items():
            #group.id = GrNr
            #groupIDs[ group.members ] = GrNr
            groupIDs[ group ] = GrNr

        SingletonIndex = np.arange(self.sample.count,)[groupIDs==-1]
        for gn, idx in zip(range(GrNr+1, GrNr+1+np.sum(groupIDs==-1)), SingletonIndex):
            groupIDs[idx] = gn
            #multirun_groups[gn] = Group(self,gn,idx)
            multirun_groups[gn] = [idx]

        return groupIDs,multirun_groups