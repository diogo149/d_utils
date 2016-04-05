import du
import du._test_utils

from du._test_utils import eq_


def _test_papply1(m):
    x = du.parallel.multiprocessing_generator.PROCESS_IDXS[-1]
    return x


def _test_papply2(ds):
    return ds.map(
        _test_papply1
    )


def test_papply():
    for x in range(5):
        l = du.dataset.from_list([{}] * 100).apply(
            du.dataset.patterns.papply,
            kwargs=dict(fn=_test_papply2, n_jobs=x),
        ).to_list(
        )
        processes = x if x > 0 else 1
        eq_(set(l), set(range(processes)))
        # there should be an equal-ish distribution of work:
        eq_(du.toolz.frequencies(l),
            du.toolz.frequencies((range(processes) * 100)[:100]))


def test_subdataset_apply():
    def foo(ds):
        return ds.mapcat_key(key="a", fn=lambda x: [x, 2 * x]).chunk()

    ds = du.dataset.from_list([{"a": 2}, {"a": 3}]).apply(
        du.dataset.patterns.subdataset_apply,
        kwargs=dict(
            fn=foo
        ))
    eq_(ds.to_list(), [{"a": [2, 4]}, {"a": [3, 6]}])
