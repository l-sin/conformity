import numpy as np
import my_functions as mf
import matplotlib.pyplot as plt
import itertools
from itertools import chain, repeat
from collections import defaultdict


#intended to replace two_point_statistics.py

def all_pairs(A,B):
    """
    given two 1d lists, A and B,
    return all possible pairings between the elements of A and the elements of B
    """
    A = sorted(list(A))
    B = sorted(list(B))
    
    return list(zip(sorted(A*len(B)), B*len(A)))


def all_combinations(list_of_lists):
    """
    Given list_of_lists = [A,B,C,...], where A=[a1,a2,...], B=[b1,b2,...],
    return all possible combinations one element from each list
    """
    flatten = lambda tp: [tp[0],*tp[1]]
    
    if len(list_of_lists) == 2:
        return all_pairs(list_of_lists[0],list_of_lists[1])
    else:
        r = all_pairs(list_of_lists[0],all_combinations(list_of_lists[1:]))
        return list(map(flatten,r))


class grid(defaultdict):
    """
    A subclass of dict with an n-dimensional grid structure, where a key is the index of a grid element,
    and the corresponding value is a list of the indices of the points within that grid element.
    """
    def __init__(self,points,spacing):
        """
        var points: list of points, where each point is a tuple of its coordinates
        var spacing: the spacing of the grid. A number, or a tuple of one spacing per dimension
        """
        super(grid, self).__init__(list)
        #inits self as an instance of defaultdict, 
        #with the default grid element being an empty list
        
        self.points = np.array(points).astype(float)
        self.spacing = spacing
        myHash = lambda dim,space: np.int32(mf.bin_hash(
                                                    dim, 
                                                    np.arange(
                                                                np.floor(min(dim)/space)*space,
                                                                np.ceil(max(dim)/space)*space+space,
                                                                space
                                                              )
                                                  ))
        
        if np.ndim(self.spacing) == len(self.points[0]):
            dims_hash = [ myHash(dim,space) for dim,space in zip( zip(*self.points), self.spacing ) ]
        elif np.ndim(self.spacing)==0:
            dims_hash = [ myHash(dim,self.spacing) for dim in zip(*self.points)]
        else:
            raise Exception(
                            "spacing must either be a number, or a tuple with dimension equal to the points"
                                )
        
        self.ndims = len(dims_hash)
        self.bounds = tuple( (min(dim),max(dim)) for dim in dims_hash )
        
        point_hash = zip( *dims_hash )
            
        for index,point in enumerate(point_hash):
            self[point].append(index)
        
    def inBound(self,elem):
        return all( el>=bnd[0] and el<=bnd[1] for el,bnd in zip(elem,self.bounds) )
        
    def neighbours(self,elem,degree=1):
        # if elem in self
        #neighs = all_combinations([ list(range(idx-1,idx+2)) for idx in elem ])
        neighs = all_combinations([ list(range(idx-degree,idx+degree+1)) for idx in elem ])
        
        return [ tuple(n) for n in neighs if self.inBound(n) ]
    
    def iter_pairs(self,chunk=False):
        """
        Generator function for all pairs between members of neighbouring grid elements.
        The default chunk=False returns pairs one at a time, while setting chunk=True returns all pairs for a given element.
        """
        for element in self.keys():
            neighs = self.neighbours(element)
            target = chain( *[t for t in map(self.get, neighs) if t is not None] )
            
            if chunk:
                yield all_pairs(self[element],target)
            else:
                for p in all_pairs(self[element],target):
                    yield p
    
    #def iter_pairs(self):
    #    """
    #    Generator function for all pairs between members of neighbouring grid elements.
    #    In the case where the space and spacing are homogeneous and isotropic, the result is
    #    guaranteed to include all pairs within the volume separated by distance <= spacing
    #    """
    #    for element in self.keys():
    #        neighs = self.neighbours(element)
    #        target = chain( *[t for t in map(self.get, neighs) if t is not None] )
    #        for p in all_pairs(self[element],target):
    #            yield p


class spherical_grid(defaultdict):
    """
    A subclass of dict with a grid structure, where a key is the index of a grid element,
    and the corresponding value is a list of the indices of the points within that grid element.
    
    Given (azimuthal, polar, radial)-coordinates on a unit sphere, 
    the grid divides the surface of the sphere such that all points angular separated by <='spacing'
    from a given point can be found within its own or its neighbouring grid elements
    """
    def __init__(self,points,spacing):
        """
        var points: list of points, given as (polar, azimuthal, radial) in ([rad],[rad],[arbitrary])
        var spacing: the spacing of the grid, given as (angular_spacing,radial_spacing)
        """
        super(spherical_grid, self).__init__(list)
        #inits self as an instance of defaultdict, 
        #with the default grid element being an empty list
        
        if len(spacing) != 2:
            raise Exception("spacing must be a tuple of (angular_spacing,radial_spacing)")
        
        pol,az,rad = zip(*points)
        az = np.array(az)
            
        self.ndims = 3
        self.points = points
        self.angular_spacing,self.radial_spacing = spacing
        
        self.radial_grid = np.arange(
                                       np.floor(min(rad)/self.radial_spacing)*self.radial_spacing,
                                       np.ceil(max(rad)/self.radial_spacing)*self.radial_spacing+self.radial_spacing,
                                       self.radial_spacing
                                     )
        self.rad_hash = np.int32(mf.bin_hash(rad, self.radial_grid))
        self.rad_bounds=(min(self.rad_hash),max(self.rad_hash))
        
        if min(az)<0 or max(az)>2*np.pi or min(pol)<0 or max(pol)>np.pi:
            raise Exception("az must be [0,2pi], pol must be [0,pi]")
        
        self.polar_spacing = np.pi/np.floor(np.pi/self.angular_spacing)
        self.polar_grid = np.arange( 0, np.pi+self.polar_spacing, self.polar_spacing )
        self.pol_hash = np.int32( mf.bin_hash(pol,self.polar_grid) )
        
        #compute delta_az (azimuthal coordinate distance) corresponding to 'spacing'
        delta_az = lambda spacing,phi: np.arccos(( np.cos(spacing)-np.cos(phi)**2 )/( np.sin(phi)**2 ) )
        
        self.az_hash = -np.ones(self.pol_hash.shape).astype('int')
        for p,phi_range in enumerate( zip(self.polar_grid[:-1],self.polar_grid[1:]) ):
            
            sel=self.pol_hash==p
            
            if p==0 or p==len(self.polar_grid)-2:
                self.az_hash[sel] = 0
            else:
                phi = phi_range[np.argmin( np.sin(phi_range) )]
                azimuthal_spacing = (2*np.pi)/np.floor((2*np.pi)/delta_az(self.angular_spacing,phi))
                azimuthal_grid = np.arange( 0, 2*np.pi+azimuthal_spacing, azimuthal_spacing )
                self.az_hash[sel] = np.int32( mf.bin_hash(az[sel],azimuthal_grid) )
        
        
        
        for index,point_hash in enumerate(zip( self.pol_hash,self.az_hash,self.rad_hash )):
            self[point_hash].append(index)
        
        
        self.f=dict()
        for p,phi_range in enumerate( zip(self.polar_grid[:-1],self.polar_grid[1:]) ):
            phi = phi_range[np.argmin( np.sin(phi_range) )]
            delta_az = np.arccos( ( np.cos(self.angular_spacing)-np.cos(phi)**2 )/( np.sin(phi)**2 ) )

            if p==0 or p==len(self.polar_grid)-2:
                self.f[p] = 1
            else:
                self.f[p] = np.floor((2*np.pi)/delta_az)
        
    def neighbours(self,elem,degree=1):
        lower_az = lambda e: e[1]*2*np.pi/self.f[e[0]]
        upper_az = lambda e: (e[1]+1)*2*np.pi/self.f[e[0]]
        overlap = lambda elem,comp: (    
                                     (lower_az(comp)<=lower_az(elem) and upper_az(comp)>=lower_az(elem)) or 
                                     (lower_az(comp)<=upper_az(elem) and upper_az(comp)>=upper_az(elem)) or
                                     (lower_az(comp)>=lower_az(elem) and upper_az(comp)<=upper_az(elem)) or
                                     (lower_az(comp)<=lower_az(elem) and upper_az(comp)>=upper_az(elem))
                                     )
        
        KTinds = all_combinations([list( range( max(elem[0]-1,0), min(elem[0]+1+1, len(self.polar_grid)-1))),
                                   list( range( max(elem[2]-1,0), min(elem[2]+1+1, self.rad_bounds[1]+1)))])
        
        neighs = list(chain(*[[(p,a,r) for a in range(int(self.f[p]))] for p,r in KTinds]))
        neighs = [ n for n in neighs if overlap(n,elem) ]

        pad = []
        for p,a,r in neighs:
            for delta_a in [-2,-1,1,2]:
                if a+delta_a<0:
                    a_da = int(self.f[p]-1)+(delta_a+1)
                elif a+delta_a>(self.f[p]-1):
                    a_da = 0+(delta_a-1)
                else:
                    a_da=a+delta_a
                
                pad_candidate = (p,a_da,r)
                if pad_candidate in self.keys() and pad_candidate not in neighs and pad_candidate not in pad:
                    pad.append(pad_candidate)
        
        neighs = [n for n in neighs if n in self.keys()]
        neighs.extend(pad)
        
        return neighs
    
    def iter_pairs(self,chunk=False):
        """
        Generator function for all pairs between members of neighbouring grid elements.
        The default chunk=False returns pairs one at a time, while setting chunk=True returns all pairs for a given element.
        """
        for element in self.keys():
            neighs = self.neighbours(element)
            target = chain( *[t for t in map(self.get, neighs) if t is not None] )
            
            if chunk:
                yield all_pairs(self[element],target)
            else:
                for p in all_pairs(self[element],target):
                    yield p
                