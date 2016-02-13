import time
from datetime import datetime, timedelta
import functools
import collections
import contextlib
from . import utils
from .utils import toolz


class TimerState(object):

    def __init__(self,
                 log_level="debug",
                 key_threshold=0,
                 key_interval=1,
                 key_frequency=0,
                 summary_frequency=1,
                 silent=False,
                 summarize=True,
                 decay=0.0):
        """
        summary_frequency - how often a summary should be printed
        key_threshold - min value of a time to log
        key_interval - how often a key is logged (in terms of number of calls)
        key_frequency - how often a key should be printed
        decay - how much less weight is given to older times (for a
                geometrically weighted rolling average)
        summarize - whether or not to print out an occasional summary
                    instead of logging every time

        NOTE: all time values are in seconds
        """
        assert 0 <= decay <= 1.0
        self.settings = utils.AttrDict(
            silent=silent,
            summarize=summarize,
            summary_frequency=summary_frequency,
            decay=decay,
            key_threshold=key_threshold,
            key_interval=key_interval,
            key_frequency=key_frequency,
            log_format=("Timer:[{key}]{{"
                        "mean={mean:.6f}, count={count}, "
                        "total={total:.2f}, summary={summary}}}")
        )
        self.set_log_level(log_level)
        self.clear()

    def clear(self):
        self.state_ = collections.defaultdict(lambda: utils.AttrDict(
            total=0.0,
            count=0,
            decayed_total=0.0,
            decayed_count=0.0,
            last_logged=-1e6,
        ))
        self.last_summary_time_ = 0
        self.to_summarize_ = set()

    def set_log_level(self, log_level):
        self.log = utils.log_level_to_log_fn[log_level]

    def keys(self):
        return self.state_.keys()

    def save_time(self,
                  key,
                  duration=None,
                  start_time=None,
                  overrides=None):
        now = time.time()

        if duration is None:
            duration = now - start_time
        if overrides is None:
            overrides = {}

        settings = utils.AttrDict(toolz.merge(self.settings, overrides))
        entry = self.state_[key]

        # update state
        decay_factor = 1 - settings.decay
        entry.decayed_total = (entry.decayed_total * decay_factor) + duration
        entry.decayed_count = (entry.decayed_count * decay_factor) + 1
        entry.total += duration
        entry.count += 1

        # log
        if not settings.silent:
            if settings.summarize:
                self.to_summarize_.add(key)
                if now - self.last_summary_time_ >= settings.summary_frequency:
                    self.last_summary_time_ = now
                    self.summarize_keys(self.to_summarize_,
                                        settings.log_format)
                    self.to_summarize_ = set()
            else:
                if (duration >= settings.key_threshold
                        and (entry.count % settings.key_interval == 0)
                        and (entry.last_logged + settings.key_frequency
                             < entry.total)):
                    entry.last_logged = entry.total
                    self.log_key(key,
                                 summary=False,
                                 log_format=settings.log_format,
                                 duration=duration)

    def mean_time(self, key):
        entry = self.state_[key]
        return entry.decayed_total / entry.decayed_count

    def summarize_keys(self, keys, log_format=None):
        if log_format is None:
            log_format = self.settings.log_format
        for key in keys:
            self.log_key(key, summary=True, log_format=log_format)

    def flush(self):
        self.summarize_keys(self.to_summarize_)

    def log_key(self, key, summary, log_format=None, duration=None):
        if log_format is None:
            log_format = self.settings.log_format
        entry = self.state_[key]
        summary = int(summary)
        format_fn = (log_format.format
                     if isinstance(log_format, str)
                     else log_format)
        msg = format_fn(
            key=key,
            mean=self.mean_time(key),
            summary=summary,
            duration=duration,
            **entry
        )
        self.log(msg)


def default_timer_params():
    return utils.config["timer"]


DEFAULT_TIMER = TimerState(**default_timer_params())


@contextlib.contextmanager
def simple_timer(key):
    start_time = time.time()
    try:
        yield
    finally:
        utils.simple_info("%s took %.6f" % (key, time.time() - start_time))


@contextlib.contextmanager
def timer(key, timer=None, **overrides):
    if timer is None:
        timer = DEFAULT_TIMER
    start_time = time.time()
    try:
        yield
    finally:
        timer.save_time(key,
                        start_time=start_time,
                        overrides=overrides)


@toolz.curry
def timed(func, **overrides):
    """
    Times the decorated function.
    """
    if hasattr(func, "func_name"):
        key = "Function[%s]" % func.func_name
    else:
        key = "Function[%s]" % func

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with timer(key, **overrides):
            return func(*args, **kwargs)
    return wrapped


class LoopTimer(object):

    """
    used to time the insides of loops (especially to estimate how much time
    is remaining in a loop)

    example:
    with LoopTimer("foo", n_iter=100) as lt:
        for _ in range(100):
            with lt.iter():
                with lt.timer("thing 1"):
                    thing1()
                with lt.timer("thing 2"):
                    thing2()
                print("Taking %f per thingy" % (lt.duration_per_iter() / 2))
    """

    def __init__(self, key, n_iter=None, summarize_after=True, **kwargs):
        self.key = key
        self.n_iter = n_iter
        self.summarize_after = summarize_after
        self.timer_state_ = TimerState(**toolz.merge(default_timer_params(),
                                                     kwargs))
        self.running_ = False
        self.iter_key_ = "%s:iter" % key
        self.log = self.timer_state_.log

    def __enter__(self):
        assert not self.running_
        self.running_ = True
        self.start_time_ = self.last_iter_end_ = time.time()
        self.iter_num_ = 1
        return self

    def duration(self):
        return time.time() - self.start_time_

    def duration_per_iter(self):
        return self.duration() / self.iter_num_

    def current_iter_duration(self):
        return time.time() - self.last_iter_end_

    def log_format(self, count, duration, mean, total, **_):
        duration_str = "%.6fs" % duration if duration is not None else ""
        if (self.n_iter is None) or (count > self.n_iter):
            return ("{key} Iter {count}:{duration}"
                    "{{mean={mean:.6f}s, total={total:.2f}s}}").format(
                        key=self.key,
                        count=count,
                        duration=duration_str,
                        mean=mean,
                        total=total
            )
        else:
            iter_remaining = self.n_iter - count
            estimated_time_left = mean * iter_remaining
            eta = datetime.now() + timedelta(seconds=estimated_time_left)
            eta_str = eta.strftime("%c")
            return ("{key} Iter {count}/{n_iter} ({pct:.1f}%):{duration}"
                    "{{mean={mean:.6f}s, total={total:.2f}s, "
                    "remaining={remaining:.0f}s, ETA={eta}}}").format(
                        key=self.key,
                        count=count,
                        n_iter=self.n_iter,
                        pct=100.0 * count / self.n_iter,
                        duration=duration_str,
                        mean=mean,
                        total=total,
                        remaining=estimated_time_left,
                        eta=eta_str,
            )

    def end_iter(self):
        """
        mark that a single loop iteration is completed - prefer using
        LoopTimer.iter in a contextmanager to not forget to call this
        """
        self.timer_state_.save_time(
            self.iter_key_,
            duration=self.current_iter_duration(),
            overrides={"log_format": self.log_format})
        self.last_iter_end_ = time.time()
        self.iter_num_ += 1

    @contextlib.contextmanager
    def iter(self):
        """
        sugar to use a contextmanager instead of calling end_iter
        """
        try:
            yield
        finally:
            self.end_iter()

    @contextlib.contextmanager
    def timer(self, key, **overrides):
        start_time = time.time()
        try:
            yield
        finally:
            self.timer_state_.save_time(key,
                                        start_time=start_time,
                                        overrides=overrides)

    def __exit__(self, type, value, tb):
        self.log("%s took %lfs", self.key, self.duration())
        if self.summarize_after:
            keys = self.timer_state_.keys()
            # commenting this out because it seems like a good thing to
            # summarize the iter_key as well
            # keys = toolz.remove(lambda x: x == self.iter_key_, keys)
            self.timer_state_.summarize_keys(keys)
        self.running_ = False
        # don't supress any exception
        return False
