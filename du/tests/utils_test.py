import doo
from doo._test_utils import equal


def test_memoize():
    x = [0]

    @doo.memoize
    def foo():
        x[0] += 1

    equal(x[0], 0)
    foo()
    equal(x[0], 1)
    foo()
    equal(x[0], 1)


def test_attr_dict():
    x = doo.AttrDict(foo=3, bar=2)
    equal(x.foo, 3)
    x.bar = 4
    equal(x.bar, 4)
