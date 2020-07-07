#print(__name__)

from galaxy_sample.samples import *
from galaxy_sample.pairs import *
from galaxy_sample.fof_group_finder import *
from galaxy_sample.yang_group_finder import *
from galaxy_sample.multirun_group_finder import *
from galaxy_sample.halo_mass_model import *
from galaxy_sample.hybrid_group_finder import *
from galaxy_sample.tinker_group_finder import *
from galaxy_sample.group_catalogue import *

"""
How namespacing works(?)
When I run 'import galaxy_sample' in __main__, Python looks up the corresponding directory, 
and runs the above under the namespace of 'galaxy_sample'. 

Therefore, 'from galaxy_sample.samples import *' imports e.g. galaxy_sample.samples.galaxy_sample 
simply as 'galaxy_sample' under the namespace of 'galaxy_sample', which would then have 
the name 'galaxy_sample.galaxy_sample' in __main__.
"""