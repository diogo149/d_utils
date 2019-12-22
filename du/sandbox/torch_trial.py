import collections
import contextlib
import datetime
import inspect
import os
import pprint

from torch.utils.tensorboard import SummaryWriter

import du
import du.torch_utils
import du.sandbox.monitor_ui


class AutoSummary(object):

    """
    smart summary class that automatically adds keys to be logged
    and understands keys to automatically add recipes

    understood prefixes:
    - _iter : the global iteration number: must be the first thing set
    - trial : keeps the name of the trial
    - time/ : keeps an average
    - loss/ : keeps min as well
    - accuracy/ : keeps max as well
    """

    def __init__(self, org_list_spaces=0):
        self.org_list_spaces = org_list_spaces
        self.fields = {}

    def setup_key(self, key, on_best=None, format=None, sample_value=None):
        assert format or sample_value

        # build a profile
        if key in self.fields:
            params = self.fields[key]
        else:
            params = du.AttrDict(key=key)
            self.fields[key] = params

        if on_best is not None:
            params.on_best = on_best
        if sample_value is not None:
            if isinstance(sample_value, str):
                params.format = "%s"
            else:
                # otherwise assume numerical
                params.format = "%.4g"

        if format is not None:
            params.format = format

        if key.startswith("loss/"):
            # lower = better
            if "best_value" not in params:
                params.best_value = None
            if "best_iter" not in params:
                params.best_iter = None
            if "update_fn" not in params:
                def update_fn():
                    if (params.best_iter is None
                        or params.value < params.best_value):
                        params.best_value = params.value
                        params.best_iter = params._iter

                        if "on_best" in params:
                            params.on_best(params)

                params.update_fn = update_fn

        elif key.startswith("accuracy/"):
            # higher = better
            if "best_value" not in params:
                params.best_value = None
            if "best_iter" not in params:
                params.best_iter = None
            if "update_fn" not in params:
                def update_fn():
                    if (params.best_iter is None
                        or params.value > params.best_value):
                        params.best_value = params.value
                        params.best_iter = params._iter

                        if "on_best" in params:
                            params.on_best(params)


                params.update_fn = update_fn
        else:
            # just keep most recent value
            assert on_best is None


    def log(self, key, value):
        if key not in self.fields:
            self.setup_key(key, sample_value=value)
        params = self.fields[key]
        # keep most recent value
        params.value = value
        # also keep the iteration it was set
        params._iter = self.fields["_iter"].value

        if "update_fn" in params:
            params.update_fn()


    def to_org_list(self):
        out_strs = []
        for f in sorted(self.fields.keys()):
            params = self.fields[f]
            if params.value is not None:
                out_strs += ["".join([" " * self.org_list_spaces,
                                      "- ",
                                      f,
                                      ": ",
                                      params.format % params.value])]

        return "\n".join(out_strs)


class TimedDataLoader(object):
    def __init__(self, key, data_loader):
        self.key = key
        self.data_loader = data_loader

    def __iter__(self):
        data_loader_iter = iter(self.data_loader)
        while True:
            with du.timer(self.key):
                try:
                    data = next(data_loader_iter)
                except StopIteration:
                    break
            yield data


def _calling_function_file():
    for frame in inspect.getouterframes(inspect.currentframe()):
        if frame[1] != __file__:
            assert frame[1].endswith(".py")
            return frame[1]

class TorchTrial(object):
    """
    features:
    - trial integration / wrapping
    - summary logging
    - tensorboard logging
    - monitor_ui logging
    - a logged timer
    - wrapping a data loader with timing
    - averaging batch stats
    - logging overall process time
    - model saving

    future features:
    - resuming training
    """

    def __init__(self,
                 name=None,
                 params=None,
                 enable_tensorboard=True,
                 enable_monitor_ui=True,
                 save_last_model=True,
                 save_best_model_metric=None,
                 org_list_spaces=2,
                 random_seed=42,
                 timer_summary_frequency=60):
        if params is None:
            params = du.AttrDict()
        if name is None:
            name = os.path.basename(_calling_function_file())[:-3]
            # add day identifier for better filtering
            name += "_" + datetime.datetime.now().strftime("%y%m%d")

        self.name = name
        self.params = params
        self.enable_tensorboard = enable_tensorboard
        self.enable_monitor_ui = enable_monitor_ui
        self.save_last_model = save_last_model
        self.save_best_model_metric = save_best_model_metric
        self.org_list_spaces = org_list_spaces

        self.trial = None
        self.summary = AutoSummary(org_list_spaces=org_list_spaces)

        # NOTE: these mutate global state
        if random_seed is not None:
            du.torch_utils.random_seed(random_seed)
        du.timer_utils.DEFAULT_TIMER.settings.summary_frequency \
            = timer_summary_frequency

        self._model = None
        self._logs = []

    def reset_iteration_log(self):
        self._iteration_log = collections.OrderedDict(
            _iter=self._iter,
            _trial=self.trial_id,
        )

    def reset_subiteration_log(self):
        # dictionary of values and counts
        self._subiteration_log = collections.defaultdict(lambda: [0, 0])

    @contextlib.contextmanager
    def run_trial(self, **kwargs):
        with du.trial.run_trial(trial_name=self.name,
                                # 1 frame for torch_trial.py
                                # 1 frame for contextlib.py
                                _run_trial_additional_frames=2,
                                **kwargs) as trial:
            self.trial = trial
            self.trial_id = "%s:%d" % (trial.trial_name, trial.iteration_num)

            # log params
            print("TorchTrial params")
            pprint.pprint(self.params)

            # setup tensorboard
            if self.enable_tensorboard:
                self.tensorboard_writer = SummaryWriter(
                    log_dir=trial.file_path("%s_tensorboard" % self.trial_id))
            else:
                self.tensorboard_writer = None

            # setup monitor_ui
            if self.enable_monitor_ui:
                self.monitor_ui_writer = du.sandbox.monitor_ui.ResultWriter(
                    dirname=trial.file_path("monitor_ui"),
                    default_settings_file=du.templates.template_path(
                        "torch_trial_monitor_ui.json"))
            else:
                self.monitor_ui_writer = None

            # setup more state
            self._iter = 1
            self.reset_iteration_log()
            self.reset_subiteration_log()
            # for the sake of ease, making these 1-indexed

            if self.save_best_model_metric is not None:
                def on_best_fn(params):
                    assert self._model is not None
                    du.torch_utils.save_model(self.trial, "best",
                                              self._model)

                    # also log the best metric
                    self.summary.log(params.key + " best",
                                     value=params.best_value)
                    self.summary.log(params.key + " best iter",
                                     value=params.best_iter)

                self.summary.setup_key(self.save_best_model_metric,
                                       on_best=on_best_fn,
                                       format="%.4g")
            yield trial

    def time_data_loader(self, key, data_loader):
        return TimedDataLoader(key, data_loader)

    def set_params(self, **kwargs):
        """
        simple wrapper for setting params
        """
        self.params.update(kwargs)

    def set_important(self, key, value):
        """
        set the value of an important param and also add it to the name
        """
        assert trial is None
        # need to replace dots because trial doesn't allow it in the name
        self.name += ("_%s-%s" % (key, value)).replace(".", "_")
        self.params[key] = value

    def register_model(self, model):
        """
        register model for some features
        e.g. model saving
        """
        print("registering model:")
        print(model)
        self._model = model


    @contextlib.contextmanager
    def logged_timer(self, key):
        with du.timer(key) as t:
            yield t
        self.log("time/%s" % key, t[0])

    def log(self, key, value):
        self._iteration_log[key] = value

    def log_subiteration(self, key, value, count):
        tmp = self._subiteration_log[key]
        tmp[0] += value * count
        tmp[1] += count

    def step(self):
        # this must be done first
        for k, (total, count) in self._subiteration_log.items():
            self.log(k, float(total) / count)

        self._logs.append(self._iteration_log)

        for k, v in self._iteration_log.items():
            self.summary.log(k, v)

        print(self.summary.to_org_list())

        if self.enable_tensorboard:
            for k, v in self._iteration_log.items():
                if "/" in k:
                    self.tensorboard_writer.add_scalar(
                        k, v, global_step=self._iter)
        if self.enable_monitor_ui:
            self.monitor_ui_writer.write(self._iteration_log)

        self._iter += 1
        self.reset_iteration_log()
        self.reset_subiteration_log()


    def done(self):
        print("DONE TorchTrial")
        self.summary.log("time/total", value=du.ps_utils.process_time())

        print(self.summary.to_org_list())

        du.io_utils.yaml_dump(self._logs, self.trial.file_path("logs.yml"))

        if self.save_last_model:
            assert self._model is not None
