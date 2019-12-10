__version__ = "0.0.1.dev0"

__title__ = "du"
__description__ = "random utils"
__uri__ = "https://github.com/diogo149/du"

__author__ = "Diogo Almeida"
__email__ = "diogo149@gmail.com"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2015 Diogo Almeida"

__all__ = """
_test_utils
cv2_utils
dicom_utils
image_processing
model
theano_utils
torch_utils
""".split()

# this needs to happen before the other imports
# so that they can use functions defined here
from .utils import *
from .timer_utils import simple_timer, timer, timed, LoopTimer

from . import data_augmentation
from . import dataset
from . import io_utils
from . import joblib_utils
from . import numpy_utils
from . import parallel
from . import performance_utils
from . import preprocessing
from . import random_utils
from . import string_utils
from . import tasks
from . import walk_utils
from . import trial
from . import unsupervised
from . import yaml_db
from . import templates
