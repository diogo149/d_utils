import doo
from doo._test_utils import equal


def test_loop_timer_doesnt_throw1():
    with doo.LoopTimer("foo") as lt:
        with lt.timer("bar"):
            pass
        for _ in range(5):
            with lt.iter():
                pass


def test_loop_timer_doesnt_throw2():
    with doo.LoopTimer("foo", n_iter=3) as lt:
        with lt.timer("bar"):
            pass
        for _ in range(5):
            with lt.iter():
                pass


def test_timer():
    with doo.timer("foo"):
        pass

    with doo.timer("foo", summary_frequency=3):
        pass

    with doo.timer("foo", silent=True):
        pass

    with doo.timer("foo", summarize=False):
        pass


def test_timer_custom_timer():
    custom_timer = doo.timer_utils.TimerState()
    with doo.timer("foo", timer=custom_timer):
        pass


def test_timed():
    @doo.timed
    def foo():
        return 3

    equal(foo(), 3)

    @doo.timed()
    def foo():
        return 4

    equal(foo(), 4)

    @doo.timed(summary_frequency=3)
    def foo():
        return 5

    equal(foo(), 5)

    @doo.timed(silent=True)
    def foo():
        return 7

    equal(foo(), 7)
