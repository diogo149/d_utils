from hyperopt import hp

import du.sandbox.hyperopt_utils
from du._test_utils import equal


def test_space_to_data():
    hyperopt_space = {
        "a": hp.lognormal("a", mu=0, sigma=2),
        "b": hp.quniform("b", low=0, high=5, q=1),
        "c": hp.quniform("c", low=0, high=5, q=1),
        "d": hp.quniform("d", low=0, high=3, q=1),
        "e": hp.quniform("e", low=1, high=5, q=1),
        "f": hp.quniform("f", low=0, high=2, q=1),
        "g": hp.uniform("g", low=0.0, high=0.7),
        "h": hp.quniform("h", low=1, high=7, q=1),
        "rest": [hp.choice("i", ["a", "b", "c", hp.randint("j", 5)])]
    }
    space_as_data = {
        'a': ('lognormal', [], {'mu': 0, 'sigma': 2}),
        'b': ('quniform', [], {'high': 5, 'q': 1, 'low': 0}),
        'c': ('quniform', [], {'high': 5, 'q': 1, 'low': 0}),
        'd': ('quniform', [], {'high': 3, 'q': 1, 'low': 0}),
        'e': ('quniform', [], {'high': 5, 'q': 1, 'low': 1}),
        'f': ('quniform', [], {'high': 2, 'q': 1, 'low': 0}),
        'g': ('uniform', [], {'high': 0.7, 'low': 0.0}),
        'h': ('quniform', [], {'high': 7, 'q': 1, 'low': 1}),
        'i': ('randint', [4], {}),
        'j': ('randint', [5], {})
    }
    equal(du.sandbox.hyperopt_utils.space_to_data(hyperopt_space),
          space_as_data)
