import copy
import time
import multiprocessing

import numpy as np
import toolz

import du
import du._test_utils
from du._test_utils import equal, assert_time, hash_equal
from nose.tools import raises


def sample_data1():
    return [{"x": 3, "y": 2},
            {"x": 5, "y": 7}]


def sample_data2():
    return [{"x": 3, "y": 2, "z": 1},
            {"x": 5, "y": 7, "z": 3}]


def test_from_list_dataset():
    x = sample_data1()
    ds = du.dataset.constructors.FromListDataset(x)
    with ds as g:
        equal(list(g), x)
    # doing it twice should work the same
    with ds as g:
        equal(list(g), x)


def test_stateless_transform_dataset():
    x = sample_data1()
    ds = du.dataset.constructors.FromListDataset(x)

    def transform(in_gen):
        for m in in_gen:
            m["z"] = m["x"] + m["y"]
            yield m

    tds = du.dataset.higher_order.StatelessTransformDataset(ds, transform)
    with tds as g:
        equal(list(g), [{"x": 3, "y": 2, "z": 5},
                        {"x": 5, "y": 7, "z": 12}])


def test_dataset_dsl_from_list():
    x = sample_data1()
    ds1 = du.dataset.constructors.FromListDataset(x)
    ds2 = du.dataset.from_list(x)
    with ds1 as g1, ds2 as g2:
        equal(list(g1), list(g2))


def test_dataset_dsl_to_list():
    x = sample_data1()
    ds = du.dataset.from_list(x)
    as_list = ds.to_list()
    with ds as g:
        equal(as_list, list(g))


def test_dataset_dsl_to_list2():
    # test that dataset doesn't remain opened after error
    ds = du.dataset.from_generator((1 / 0 for _ in range(2)))
    try:
        ds.to_list()
        assert False
    except ZeroDivisionError:
        pass
    assert not ds.dataset.opened_


def test_dataset_dsl_to_list3():
    # test that dataset doesn't remain opened after error
    ds = du.dataset.promise().map(lambda x: x)
    try:
        ds.to_list()
        assert False
    except AssertionError:
        pass
    """
    the issue with this is an exception occurs during __enter__,
    __exit__ doesn't get called
    """
    # NOTE: this means that the state was NOT properly cleaned up
    assert ds.dataset.opened_


def test_dataset_dsl_mapcat():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.mapcat(lambda x: [{"x": x["y"], "y": x["x"]}])
    with ds3 as g:
        equal(list(g), [{"x": 2, "y": 3},
                        {"x": 7, "y": 5}])
    ds4 = ds2.mapcat(lambda x: [x, x])
    with ds4 as g:
        equal(list(g), [x[0], x[0], x[1], x[1]])


def test_dataset_dsl_mapcat2():
    x = sample_data1()
    ds = du.dataset.from_list(x)
    ds2 = ds.mapcat(key="x", out="x", fn=lambda x: [x, -x])
    equal(ds2.to_list(), [{"x": 3, "y": 2},
                          {"x": -3, "y": 2},
                          {"x": 5, "y": 7},
                          {"x": -5, "y": 7}])


def test_dataset_dsl_mapcat3():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.mapcat(key="x", out="z", fn=lambda x: [x, -x])
    equal(ds3.to_list(), [{"x": 3, "y": 2, "z": 3},
                          {"x": 3, "y": 2, "z": -3},
                          {"x": 5, "y": 7, "z": 5},
                          {"x": 5, "y": 7, "z": -5}])


def test_dataset_dsl_mapcat_key():
    x = sample_data1()
    ds = du.dataset.from_list(x)
    ds2 = ds.mapcat_key(key="x", fn=lambda x: [x, -x])
    equal(ds2.to_list(), [{"x": 3, "y": 2},
                          {"x": -3, "y": 2},
                          {"x": 5, "y": 7},
                          {"x": -5, "y": 7}])


def test_dataset_dsl_mapcat_key_unicode():
    x = sample_data1()
    ds = du.dataset.from_list(x)
    ds2 = ds.mapcat_key(key=u"x", fn=lambda x: [x, -x])
    equal(ds2.to_list(), [{"x": 3, "y": 2},
                          {u"x": -3, "y": 2},
                          {"x": 5, u"y": 7},
                          {u"x": -5, u"y": 7}])


def test_dataset_dsl_map():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.map(lambda x: {"x": x["y"], "y": x["x"]})
    equal(ds3.to_list(), [{"x": 2, "y": 3},
                          {"x": 7, "y": 5}])
    ds4 = ds2.map(lambda x: x)
    equal(ds4.to_list(), x)


def test_dataset_dsl_map2():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.map(key="x", out="x", fn=lambda x: x * 2)
    equal(ds3.to_list(), [{"x": 6, "y": 2},
                          {"x": 10, "y": 7}])


def test_dataset_dsl_map3():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.map(out="z", fn=lambda x: x["x"] + x["y"])
    equal(ds3.to_list(), [{"x": 3, "y": 2, "z": 5},
                          {"x": 5, "y": 7, "z": 12}])


def test_dataset_dsl_map4():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.map(key="x", out="z", fn=lambda x: 3 * x)
    equal(ds3.to_list(), [{"x": 3, "y": 2, "z": 9},
                          {"x": 5, "y": 7, "z": 15}])


def test_dataset_dsl_map5():
    equal(du.dataset.from_list([1, 2, 3]).map(lambda x: x + 1).to_list(),
          [2, 3, 4])


def test_dataset_dsl_map_key():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.map_key(key="x", fn=lambda x: x * 2)
    equal(ds3.to_list(), [{"x": 6, "y": 2},
                          {"x": 10, "y": 7}])


def test_dataset_dsl_map_tuple_key():
    x = sample_data1()
    ds = du.dataset.from_list(x)
    ds2 = ds.map(key=("x", "y"), out="z", fn=lambda x, y: x * y)
    equal(ds2.to_list(), [{"x": 3, "y": 2, "z": 6},
                          {"x": 5, "y": 7, "z": 35}])


def test_dataset_dsl_select_tuple():
    x = sample_data2()
    ds = du.dataset.from_list(x).select_keys(("x", "y"))
    equal(ds.to_list(), [{"x": 3, "y": 2},
                         {"x": 5, "y": 7}])


def test_dataset_dsl_rename_tuple():
    x = sample_data1()
    ds = du.dataset.from_list(x).rename(("x", "y"), ["x_new", "y_new"])
    equal(ds.to_list(), [{"x_new": 3, "y_new": 2},
                         {"x_new": 5, "y_new": 7}])


def test_dataset_dsl_map_each_key1():
    x = sample_data1()
    ds2 = du.dataset.from_list(x).assoc_constant("z", 5)
    ds3 = ds2.map_each_key(keys=["x", "y"], fn=lambda x: x * 2)
    equal(ds3.to_list(), [{"x": 6, "y": 4, "z": 5},
                          {"x": 10, "y": 14, "z": 5}])


@raises(TypeError)
def test_dataset_dsl_map_each_key2():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.map_each_key(fn=lambda x: x * 2)
    ds3.to_list()


@raises(TypeError)
def test_dataset_dsl_map_each_key3():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.map_each_key(key=['x', 'y'], fn=lambda x: x * 2)
    ds3.to_list()


def test_dataset_dsl_filter():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.filter(lambda x: True)
    with ds3 as g:
        equal(list(g), x)
    ds4 = ds2.filter(lambda x: x["x"] == 3)
    with ds4 as g:
        equal(list(g), [x[0]])


def test_dataset_dsl_filter2():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.filter(key="x", fn=lambda x: x == 5)
    equal(ds3.to_list(), [x[1]])


def test_dataset_dsl_remove():
    x = sample_data1()
    ds = du.dataset.from_list(x)
    equal(ds.remove(lambda x: True).to_list(), [])
    equal(ds.remove(lambda x: False).to_list(), x)
    equal(ds.remove(lambda x: x["x"] == 3).to_list(), [x[1]])
    equal(ds.remove(lambda x: x["x"] != 3).to_list(), [x[0]])


def test_dataset_dsl_remove2():
    x = sample_data1()
    ds = du.dataset.from_list(x)
    equal(ds.remove(key="x", fn=lambda x: x == 5).to_list(), [x[0]])


def test_dataset_dsl_select_keys():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.select_keys(["y"])
    with ds3 as g:
        equal(list(g), [{"y": 2},
                        {"y": 7}])


def test_dataset_dsl_dissoc():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.dissoc("x")
    with ds3 as g:
        equal(list(g), [{"y": 2},
                        {"y": 7}])
    ds4 = ds2.dissoc(["x"])
    with ds4 as g:
        equal(list(g), [{"y": 2},
                        {"y": 7}])


def test_dataset_dsl_rename():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.rename("x", "z")
    with ds3 as g:
        equal(list(g), [{"z": 3, "y": 2},
                        {"z": 5, "y": 7}])


def test_dataset_dsl_zip_assoc():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.zip_assoc("z", range(100))
    with ds3 as g:
        equal(list(g), [toolz.assoc(x[0], "z", 0),
                        toolz.assoc(x[1], "z", 1)])
    ds4 = ds2.zip_assoc("z", range(1))
    with ds4 as g:
        equal(list(g), [toolz.assoc(x[0], "z", 0)])
    ds5 = ds2.zip_assoc("z", [])
    with ds5 as g:
        equal(list(g), [])


def test_dataset_dsl_zip_merge():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.zip_merge([{"z": 0}, {"z": 1}])
    with ds3 as g:
        equal(list(g), [toolz.assoc(x[0], "z", 0),
                        toolz.assoc(x[1], "z", 1)])
    ds4 = ds2.zip_merge([{"z": 100}])
    with ds4 as g:
        equal(list(g), [toolz.assoc(x[0], "z", 100)])
    ds5 = ds2.zip_merge([])
    with ds5 as g:
        equal(list(g), [])
    ds6 = ds2.zip_merge([{"x": 42}])
    with ds6 as g:
        equal(list(g), [{"x": 42, "y": 2}])


def test_dataset_dsl_zip_merge_with():

    def plus(xs):
        return sum(xs)

    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.zip_merge_with(plus, [{"z": 0}, {"z": 1}])
    with ds3 as g:
        equal(list(g), [toolz.assoc(x[0], "z", 0),
                        toolz.assoc(x[1], "z", 1)])
    ds4 = ds2.zip_merge_with(plus, [{"z": 100}])
    with ds4 as g:
        equal(list(g), [toolz.assoc(x[0], "z", 100)])
    ds5 = ds2.zip_merge_with(plus, [])
    with ds5 as g:
        equal(list(g), [])
    ds6 = ds2.zip_merge_with(plus, [{"x": 42}])
    with ds6 as g:
        equal(list(g), [{"x": 45, "y": 2}])


def test_dataset_dsl_do():
    state = {"x": 0}

    def update_x(m):
        state["x"] += m["x"]

    x = sample_data1()
    ds = du.dataset.from_list(x)
    ds1 = ds.do(update_x)
    with ds1 as g:
        equal(list(g), x)
        equal(state["x"], 8)


def test_dataset_dsl_do2():
    state = {"x": 0}

    def update_x(x):
        state["x"] += x

    x = sample_data1()
    ds = du.dataset.from_list(x)
    ds1 = ds.do(update_x, "x")
    with ds1 as g:
        equal(list(g), x)
        equal(state["x"], 8)


def test_dataset_dsl_copy():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.assoc_constant("z", 42)
    ds4 = ds3.map(key="x", out="w", fn=lambda x: 3 * x)
    with ds4.copy() as g1:
        with ds4 as g2:
            equal(list(g1), list(g2))
    ds5 = ds3.copy().map(key="x",
                         out="w",
                         fn=lambda x: 3 * x)
    with ds4 as g1:
        with ds5 as g2:
            equal(list(g1), list(g2))


def test_dataset_dsl_assoc_constant():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.assoc_constant("z", 42)
    with ds3 as g:
        equal(list(g), [{"x": 3, "y": 2, "z": 42},
                        {"x": 5, "y": 7, "z": 42}])


def test_dataset_dsl_take():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.take(1)
    with ds3 as g:
        equal(list(g), x[:1])
    ds4 = ds2.take(2)
    with ds4 as g:
        equal(list(g), x)
    ds5 = ds2.take(2)
    with ds5 as g:
        equal(list(g), x)


def test_dataset_dsl_repeat_each():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.repeat_each(1)
    equal(ds3.to_list(), ds2.to_list())
    ds4 = ds2.repeat_each(2)
    equal(ds4.to_list(), [x[0], x[0], x[1], x[1]])


def test_dataset_dsl_chunk():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.chunk(chunk_size=2)
    with ds3 as g:
        chunk_size_2 = list(g)
        equal(chunk_size_2, [{"x": [3, 5], "y": [2, 7]}])
    ds4 = ds2.chunk(chunk_size=-1)
    with ds4 as g:
        equal(list(g), chunk_size_2)
    ds5 = ds2.chunk(chunk_size=1)
    with ds5 as g:
        equal(list(g), [{"x": [3], "y": [2]},
                        {"x": [5], "y": [7]}])


def test_dataset_dsl_chunk2():
    xs = [1, 2, 3, 4, 5]
    ds = du.dataset.from_list([dict(x=x) for x in xs])
    ds2 = ds.chunk(chunk_size=2).dechunk()
    equal(map(lambda x: x["x"], ds2.to_list()),
          [1, 2, 3, 4])
    ds2 = ds.chunk(chunk_size=2, drop_remainder=False).dechunk()
    equal(map(lambda x: x["x"], ds2.to_list()),
          [1, 2, 3, 4, 5])


def test_dataset_dsl_dechunk():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.chunk(2).dechunk()
    with ds3 as g:
        equal(list(g), x)


def test_dataset_dsl_sort1():
    def return_sorted(l, **kwargs):
        return du.dataset.from_list(l).sort(**kwargs).to_list()
    x = [{"x": 2},
         {"x": 3},
         {"x": 3},
         {"x": 5}]
    equal(x, return_sorted(x, key="x"))
    equal(x[::-1], return_sorted(x, key="x", reverse=True))
    equal(x[::-1], return_sorted(x, key="x", fn=lambda x: -x))
    equal([x[1], x[2], x[0], x[3]],
          return_sorted(x, key="x", fn=lambda x: -100 if x == 3 else x))


def test_dataset_dsl_sort2():
    # test with an additional key
    def return_sorted(l, **kwargs):
        return du.dataset.from_list(l).sort(**kwargs).to_list()
    x = [{"x": 2, "y": 2},
         {"x": 3, "y": 1},
         {"x": 5, "y": 4}]
    equal(x, return_sorted(x, key="x"))
    equal(x[::-1], return_sorted(x, key="x", reverse=True))
    equal(x[::-1], return_sorted(x, key="x", fn=lambda x: -x))
    equal([x[1], x[0], x[2]],
          return_sorted(x, key="x", fn=lambda x: -100 if x == 3 else x))


def test_dataset_dsl_numpy_chunk():
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.numpy_chunk(["x", "y"], chunk_size=2)
    with ds3 as g:
        chunk_size_2 = list(g)
        hash_equal(
            chunk_size_2, [{"x": np.array([3, 5]), "y": np.array([2, 7])}])
    ds4 = ds2.numpy_chunk(["x", "y"], chunk_size=-1)
    with ds4 as g:
        hash_equal(list(g), chunk_size_2)
    ds5 = ds2.numpy_chunk(["x", "y"], chunk_size=1)
    with ds5 as g:
        hash_equal(list(g), [{"x": np.array([3]), "y": np.array([2])},
                             {"x": np.array([5]), "y": np.array([7])}])
    ds6 = ds2.numpy_chunk(["x"], chunk_size=2)
    with ds6 as g:
        chunk_size_2 = list(g)
        hash_equal(chunk_size_2, [{"x": np.array([3, 5]), "y": [2, 7]}])


def test_dataset_dsl_random_sample():
    rng = np.random.RandomState(42)
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.random_sample(rng=rng)
    with ds3 as g:
        equal(toolz.thread_last(g,
                                (toolz.take, 50),
                                (map, lambda x: x["x"]),
                                toolz.frequencies),
              {3: 23, 5: 27})
    ds4 = ds2.random_sample(rng=rng, count=10)
    with ds4 as g:
        equal(toolz.thread_last(g,
                                (toolz.take, 50),
                                (map, lambda x: x["x"]),
                                toolz.frequencies),
              {3: 6, 5: 4})


def test_dataset_dsl_random_sample_empty():
    rng = np.random.RandomState(42)
    ds2 = du.dataset.from_list([])
    ds3 = ds2.random_sample(rng=rng)
    equal(ds3.to_list(), [])


def test_dataset_dsl_tic():
    pre_run = time.time()
    ds = du.dataset.from_list([{}]).tic(out="foobar")
    pre_run2 = time.time()
    l = ds.to_list()
    post_run = time.time()
    tic_time = l[0]["foobar"]
    assert pre_run < pre_run2 < tic_time < post_run


def test_dataset_dsl_cache():
    with du.io_utils.NamedTemporaryDirectory() as dirname:
        m = {}
        x = sample_data1()
        ds2 = du.dataset.from_list(x)

        def update_m(x):
            m["x"] += 1
            return x

        m["x"] = 0
        ds3 = ds2.map_key(
            fn=update_m,
            key="x",
        ).cache(
            dirname=dirname,
            backend="joblib",
        )
        equal(m["x"], 0)
        with ds3 as g:
            equal(list(g), x)
            equal(m["x"], 2)
        # test that cache is idempotent
        with ds3 as g:
            equal(list(g), x)
            equal(m["x"], 2)
        ds4 = ds2.map(update_m, "x", "x").cache(
            backend="joblib",
            dirname=dirname,
        )
        with ds4 as g:
            equal(list(g), x)
            equal(m["x"], 2)
        ds5 = ds2.map(update_m, "x", "x").cache(
            backend="joblib",
            dirname=dirname,
            version=42,
        )
        with ds5 as g:
            equal(list(g), x)
            equal(m["x"], 4)
        ds7 = ds2.map(update_m, "x", "x").cache(
            backend="joblib",
            dirname=dirname,
            version=43,
            eager=True,
        )
        equal(m["x"], 6)
        with ds7 as g:
            equal(list(g), x)
            equal(m["x"], 6)


def test_dataset_dsl_doall():
    x = sample_data1()
    equal(du.dataset.from_list(x).doall().to_list(), [])
    equal(du.dataset.from_list(x).doall(discard=False).to_list(), x)


def test_dataset_dsl_repeat_all():
    res = du.dataset.from_list(
        [{"x": i} for i in [1, 2, 3]]
    ).repeat_all().take(100).to_list()
    equal(res,
          [{"x": 1}, {"x": 2}, {"x": 3}] * 33 + [{"x": 1}])


def test_dataset_dsl_apply():
    x = sample_data1()

    def inner(ds):
        return ds.map_key(lambda x: x * 2, "x")

    equal(du.dataset.from_list(x).map_key(lambda x: x * 2, "x").to_list(),
          du.dataset.from_list(x).apply(inner).to_list())


def test_random_sample_generators():
    rng = np.random.RandomState(42)
    equal(list(du.dataset.higher_order.random_sample_generators(
        [(x for x in [1, 2, 3]),
         (x for x in [4, 5])],
        weights=[0.99, 0.01],
        rng=rng)),
        [1, 2, 3, 4, 5])


def test_multi_dataset():
    rng = np.random.RandomState(42)
    x = sample_data1()
    ds1 = du.dataset.from_list(x)
    ds2 = ds1.map(lambda x: -x, "x", "x")
    mds = du.dataset.higher_order.MultiDataset(
        [copy.deepcopy(ds1), ds2],
        strategy="random",
        strategy_opts=dict(rng=rng,
                           weights=[0.99, 0.01]))
    with mds as g:
        equal([m["x"] for m in g], [x[0]["x"],
                                    x[1]["x"],
                                    -x[0]["x"],
                                    -x[1]["x"]])


def test_dataset_dsl_promise():
    x = sample_data1()
    base_ds = du.dataset.promise()
    ds = base_ds.map_key(key="x", fn=lambda x: x * 2)
    # opening a non-realized promise should error
    try:
        ds.copy().to_list()
        assert False
    except AssertionError:
        pass
    # realizing a promise should work
    print ds.dataset.opened_
    ds2 = du.dataset.from_list(x)
    base_ds.dataset.deliver(ds2)
    equal(ds.to_list(), [{"x": 6, "y": 2},
                         {"x": 10, "y": 7}])
    # should not be able to realize twice
    try:
        base_ds.dataset.deliver(ds2)
        assert False
    except AssertionError:
        pass


def test_dataset_dsl_to_fn():
    x = sample_data1()
    ds = du.dataset.from_list(x)
    ds2 = ds.apply(
        du.dataset.promise().map_key(key="x", fn=lambda x: x * 2).to_fn()
    )
    equal(ds2.to_list(), [{"x": 6, "y": 2},
                          {"x": 10, "y": 7}])


def slow_computation(x, delay=None):
    if delay:
        du.info("Start sleeping")
        time.sleep(delay)
        du.info("Done sleeping")
    x["x"] *= 42
    x["y"] *= 2
    return x


def slow_computation_list(x, delay=None):
    return [slow_computation(x, delay)]


@du._test_utils.slow
def test_dataset_dsl_to_ther_process():
    delay = 0.5
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.map(
        slow_computation, kwargs=dict(delay=delay)
    ).to_other_process()
    with assert_time(delay * 2, delay * 2.5):
        with ds3 as g:
            equal(list(g), map(slow_computation, x))
    with ds3 as g:
        # other process should be computing while this process waits
        time.sleep(delay * 2.5)
        with assert_time(0, delay * 0.2):
            equal(list(g), map(slow_computation, x))


@du._test_utils.slow
def test_dataset_dsl_pmapcat():
    delay = 1
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.pmapcat(slow_computation_list,
                      backend="joblib",
                      joblib_n_jobs=2,
                      kwargs=dict(delay=delay))
    with assert_time(delay, delay * 1.5):
        with ds3 as g:
            equal(list(g), map(slow_computation, x))
    pool = multiprocessing.Pool(processes=2)
    try:
        # with map
        ds4 = ds2.pmapcat(slow_computation_list,
                          backend="pool",
                          pool=pool,
                          pool_fn="map",
                          kwargs=dict(delay=delay))
        with assert_time(delay, delay * 1.5):
            with ds4 as g:
                equal(list(g), map(slow_computation, x))
        # with imap
        ds4 = ds2.pmapcat(slow_computation_list,
                          backend="pool",
                          pool=pool,
                          pool_fn="imap",
                          kwargs=dict(delay=delay))
        with assert_time(delay, delay * 1.5):
            with ds4 as g:
                equal(list(g), map(slow_computation, x))
    finally:
        pool.close()


@du._test_utils.slow
def test_dataset_dsl_threaded_reader():
    delay = 0.5
    x = sample_data1()
    ds2 = du.dataset.from_list(x)
    ds3 = ds2.map(
        slow_computation, kwargs=dict(delay=delay)
    ).to_threaded_reader()
    with assert_time(delay, delay * 1.5):
        with ds3 as g:
            du.info("Sleeping")
            time.sleep(delay)
            du.info("Awake")
            equal(g.next(), slow_computation(x[0]))
