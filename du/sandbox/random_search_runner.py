import collections
import functools
import pandas as pd
import du
from du.sandbox import hyperopt_utils


def repeat_random_search(jsonl_filename,
                         hyperparameters_fn,
                         targets_fn):
    """
    hyperparameters_fn:
    function that takes no arguments and returns a dictionary from
    hyperparameter name to value

    targets_fn:
    takes in hyperparameters from hyperparameters_fn
    """
    results = []
    try:
        while True:
            hyperparameters = hyperparameters_fn()
            targets = targets_fn(**hyperparameters)
            result = dict(
                hyperparameters=hyperparameters,
                targets=targets,
            )
            du.io_utils.jsonl_append(result, jsonl_filename)
            results.append(result)
    except KeyboardInterrupt:
        pass
    return results


def write_html_report(jsonl_filename,
                      html_filename,
                      hyperparameters=None,
                      target="loss",
                      goal="minimize",
                      **kwargs):
    """
    hyperparameters:
    if None, use all of them
    """
    data = du.io_utils.jsonl_load(jsonl_filename)

    df_data = collections.defaultdict(list)

    if hyperparameters is None:
        # gather set of all hyperparameters that all rows have
        hyperparameters = functools.reduce(
            set.intersection,
            [set(row["hyperparameters"].keys()) for row in data])

    for row in data:
        loss = row["targets"][target]
        if goal == "maximize":
            loss = -loss
        df_data["loss"].append(loss)
        for hp in hyperparameters:
            df_data[hp].append(row["hyperparameters"][hp])

    df = pd.DataFrame(df_data)

    html = hyperopt_utils.html_hyperopt_report(trials_df=df, **kwargs)
    with open(html_filename, "w") as f:
        f.write(html)
