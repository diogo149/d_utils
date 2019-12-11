from __future__ import absolute_import

# imports are implicit in python 3
# import base
# import constructors
# import higher_order
# import dsl
from .dsl import (DatasetDSL,
                 from_list,
                 from_generator,
                 from_generator_fn,
                 from_joblib_dir,
                 promise,
                 multi_dataset)

from . import extras
from . import patterns
