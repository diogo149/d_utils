import numpy as np

from .. import utils, random_utils


def _random_split(in_datamap, validation_ratio, validation_set, rng):
    validation_datamap = rng.rand() < validation_ratio
    return validation_datamap == validation_set


def random_split(validation_ratio, validation_set=False, rng=None):
    # don't use a default rng here, since the behavior you generally
    # want is for this to be deterministic
    # if you do not want to provide an rng, pass in np.random
    assert rng is not None
    return utils.partial(_random_split,
                         validation_ratio=validation_ratio,
                         validation_set=validation_set,
                         rng=rng)


def _validation_split(validation_ratio,
                      is_validation_set,
                      salt,
                      *in_vals):
    rand = random_utils.anything_to_random_state(in_vals, salt=salt).rand()
    validation_datamap = rand < validation_ratio
    return validation_datamap == is_validation_set


def validation_split(validation_ratio,
                     is_validation_set,
                     salt=0):
    """
    validation_ratio:
    ratio of elements to be part of validation set

    is_validation_set:
    if true, returns true if random number is less than validation_ratio
    (signifying that the element is a member of the validation set) and returns
    false otherwise. does the opposite if false (returning true for elements
    not in the validation set)

    salt:
    added to random seed to allow different random numbers generated for the
    same values not to be correlated
    """
    assert 0 <= validation_ratio <= 1
    assert isinstance(salt, int)
    return utils.partial(_validation_split,
                         validation_ratio,
                         is_validation_set,
                         salt)


def _log_as(in_datamap, format_string):
    utils.info(format_string.format(**in_datamap))


def log_as(format_string):
    return utils.partial(_log_as,
                         format_string=format_string)
