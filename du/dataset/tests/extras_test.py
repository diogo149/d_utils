import du
import du._test_utils

from du._test_utils import eq_


def sample_data1():
    return [{"x": 3, "y": 2},
            {"x": 5, "y": 7}]


def test_validation_split():
    xs = [1, 2, 3, 1, 2, 3, 2, 3, 2]
    data = [dict(x=x) for x in xs]
    ds = du.dataset.from_list(data).filter(
        key="x",
        fn=du.dataset.extras.validation_split(
            validation_ratio=0.5,
            is_validation_set=True,
        ))
    eq_(du.toolz.frequencies(map(lambda x: x["x"], ds.to_list())),
        {1: 2, 2: 4})
