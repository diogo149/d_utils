import numbers

import numpy as np
import theano
import theano.tensor as T

fX = theano.config.floatX
np_fX = np.dtype(fX)


def as_fX(x):
    """
    convert input to value with type floatX
    """
    if isinstance(x, (float, numbers.Integral)):
        return np.array(x, dtype=fX)
    elif isinstance(x, np.ndarray):
        if x.dtype == np_fX:
            # don't make a copy if not necessary
            return x
        else:
            return x.astype(fX)
    else:
        # assume theano variable
        if x.dtype == fX:
            return x
        else:
            return x.astype(fX)


def is_nonshared_variable(x):
    return isinstance(x, theano.gof.graph.Variable)


def is_shared_variable(x):
    return isinstance(x, theano.compile.sharedvalue.SharedVariable)


def is_variable(x):
    return is_nonshared_variable(x) or is_shared_variable(x)
