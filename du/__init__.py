__version__ = "0.0.1.dev0"

__title__ = "du"
__description__ = "random utils"
__uri__ = "https://github.com/diogo149/du"

__author__ = "Diogo Almeida"
__email__ = "diogo149@gmai.com"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2015 Diogo Almeida"

__all__ = """
_test_utils
cv2_utils
dicom_utils
image_processing
model
theano_utils
""".split()

# this needs to happen before the other imports
# so that they can use functions defined here
from .utils import *
from .timer_utils import simple_timer, timer, timed, LoopTimer

import data_augmentation
import dataset
import io_utils
import joblib_utils
import numpy_utils
import parallel
import performance_utils
import preprocessing
import random_utils
import walk_utils
import trial
import unsupervised
import yaml_db
