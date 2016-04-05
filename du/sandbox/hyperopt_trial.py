import os

import hyperopt

from . import hyperopt_utils
from .. import joblib_utils
from .. import trial
from .. import utils


def _hyperopt_inner_trial(params, trial_function, **kwargs):
    """
    run trial_function with the given hyperopt params

    in a separate top-level function to be pickle-able
    """
    return trial.run_trial_function(trial_function, args=(params,), **kwargs)


def hyperopt_trial(trial_name,
                   trial_function,
                   hyperopt_space,
                   hyperopt_evals,
                   snippets,
                   **kwargs):
    """
    allows running a hyperopt trial, with individual hyperopt runs in the
    trial directory

    this avoids a nasty bug with trials reading the most recent version of
    a file instead of reading the version of the file when beginning running
    hyperopt

    trial_function:
    function that takes in a TrialState instance and hyperopt params as input
    """
    exp_key = "exp1"

    with trial.run_trial(trial_name=trial_name,
                         snippets=snippets,
                         **kwargs) as outer_trial:
        inner_trials_dir = outer_trial.file_path("hyperopt_trials")

        # TODO work on mongo version
        # trials = hyperopt.mongoexp.MongoTrials(mongo_url,
        #                                        exp_key=exp_key)
        hyperopt_trials = hyperopt.Trials(exp_key=exp_key)

        # generate new snippets that use the copied versions of the files
        new_snippets = [[snippet_name,
                         outer_trial.src_path("%s.py" % snippet_name)]
                        for snippet_name, _ in snippets]
        for _, snippet_path in new_snippets:
            assert os.path.exists(snippet_path)

        loss_fn = utils.partial(_hyperopt_inner_trial,
                                trial_function=trial_function,
                                trial_name="inner_trial",
                                snippets=new_snippets,
                                trials_dir=inner_trials_dir)

        # TODO maybe put this in try finally
        best = hyperopt.fmin(fn=loss_fn,
                             space=hyperopt_space,
                             trials=hyperopt_trials,
                             # can also use hyperopt.rand.suggest
                             algo=hyperopt.tpe.suggest,
                             max_evals=hyperopt_evals)
        utils.info("Best parameters: %s", best)

        trials_df = hyperopt_utils.trials_to_df(hyperopt_trials)

        # save final trials and trials_df (just to be sure)
        joblib_utils.dump_dir(hyperopt_trials,
                              outer_trial.file_path("hyperopt_trials"))
        joblib_utils.dump_dir(trials_df,
                              outer_trial.file_path("trials_df"))

        # write report
        html = hyperopt_utils.html_hyperopt_report(trials_df)
        with open(outer_trial.file_path("hyperopt_report.html"), "w") as f:
            f.write(html)

        return trials_df
