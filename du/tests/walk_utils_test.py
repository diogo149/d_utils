from du._test_utils import equal, raises
import du.walk_utils


def find_steps(walk_fn, data):
    steps = []

    def prewalk_fn(x):
        steps.append(("pre", x))
        return x

    def postwalk_fn(x):
        steps.append(("post", x))
        return x

    walk_fn(data,
            prewalk_fn,
            postwalk_fn)
    return steps


def test_walk_identity():
    data = [(1, 2, 3)]
    equal(du.walk_utils.walk(data),
          data)
    equal(du.walk_utils.walk(data, lambda x: x, lambda x: x),
          data)


def test_walk():
    equal(find_steps(du.walk_utils.walk, [(1, 2, 3)]),
          [("pre", [(1, 2, 3)]),
           ("pre", (1, 2, 3)),
           ("pre", 1),
           ("post", 1),
           ("pre", 2),
           ("post", 2),
           ("pre", 3),
           ("post", 3),
           ("post", (1, 2, 3)),
           ("post", [(1, 2, 3)])])


def test_collection_walk():
    data = [(1, 2, 3)]
    # NOTE: they will not return the same steps for maps and sets
    equal(find_steps(du.walk_utils.collection_walk, data),
          find_steps(du.walk_utils.walk, data))


def test_walk_numpy():
    import numpy as np

    def numpy_data_fn(x):
        if isinstance(x, np.ndarray):
            return x.shape
        else:
            return x

    data = ({"bar": np.random.randn(500)},
            [(3, 4, 5, np.random.rand(3, 4))],
            set([(3, 4)]))
    equal(du.walk_utils.walk(data, prewalk_fn=numpy_data_fn),
          ({"bar": (500,)},
           [(3, 4, 5, (3, 4))],
           {(3, 4)}))


@raises(du.walk_utils.CyclicWalkException)
def test_walk_cyclic1():
    x = []
    x.append(x)
    du.walk_utils.walk(x)


@raises(du.walk_utils.CyclicWalkException)
def test_walk_cyclic2():
    y = []
    y.append([[[[[y]]]]])
    du.walk_utils.walk(y)


def find_steps_donewalkingexception(walk_fn, data, exceptions):
    steps = []

    def prewalk_fn(x):
        steps.append(("pre", x))
        if x in exceptions:
            raise du.walk_utils.DoneWalkingException(x)
        return x

    def postwalk_fn(x):
        steps.append(("post", x))
        return x

    walk_fn(data,
            prewalk_fn,
            postwalk_fn)
    return steps


def test_walk_donewalkingexception():
    equal(find_steps_donewalkingexception(du.walk_utils.walk,
                                          [(1, 2, 3)],
                                          [(1, 2, 3)]),
          [("pre", [(1, 2, 3)]),
           ("pre", (1, 2, 3)),
           ("post", [(1, 2, 3)])])

    equal(find_steps_donewalkingexception(du.walk_utils.walk,
                                          [(1, 2, 3)],
                                          [1, 2, 3]),
          [("pre", [(1, 2, 3)]),
           ("pre", (1, 2, 3)),
           ("pre", 1),
           ("pre", 2),
           ("pre", 3),
           ("post", (1, 2, 3)),
           ("post", [(1, 2, 3)])])
