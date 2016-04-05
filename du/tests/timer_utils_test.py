import du
from du._test_utils import equal


def test_loop_timer_doesnt_throw1():
    with du.LoopTimer("foo") as lt:
        with lt.timer("bar"):
            pass
        for _ in range(5):
            with lt.iter():
                pass


def test_loop_timer_doesnt_throw2():
    with du.LoopTimer("foo", n_iter=3) as lt:
        with lt.timer("bar"):
            pass
        for _ in range(5):
            with lt.iter():
                pass


def test_timer():
    with du.timer("foo"):
        pass

    with du.timer("foo", summary_frequency=3):
        pass

    with du.timer("foo", silent=True):
        pass

    with du.timer("foo", summarize=False):
        pass


def test_timer_custom_timer():
    custom_timer = du.timer_utils.TimerState()
    with du.timer("foo", timer=custom_timer):
        pass


def test_timed():
    @du.timed
    def foo():
        return 3

    equal(foo(), 3)

    @du.timed()
    def foo():
        return 4

    equal(foo(), 4)

    @du.timed(summary_frequency=3)
    def foo():
        return 5

    equal(foo(), 5)

    @du.timed(silent=True)
    def foo():
        return 7

    equal(foo(), 7)
