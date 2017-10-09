import os
import itertools
import StringIO
import urllib
import base64
import numpy as np
import pandas as pd
import hyperopt

from .. import numpy_utils
from .. import walk_utils
from .. import utils


def trial_to_dict(trial):
    out = {}
    result = trial["result"]
    if result["status"] == hyperopt.STATUS_OK:
        out["loss"] = result["loss"]
    for k, v in trial["misc"]["vals"].items():
        assert isinstance(v, list)
        assert len(v) in [0, 1]
        if v:
            out[k] = v[0]
    return out


def trials_to_df(trials):
    return pd.DataFrame(map(trial_to_dict, trials.trials))


def space_to_data(space):
    """
    converts hyperparameter space into data
    """

    res = {}

    def walk_fn(x):
        if isinstance(x, hyperopt.pyll.base.Apply):
            if x.name == "hyperopt_param":
                name_node = x.pos_args[0]
                name = name_node._obj
                apply_node = x.pos_args[1]
                distribution_name = apply_node.name
                distribution_args = [literal._obj
                                     for literal in apply_node.pos_args]
                distribution_kwargs = {arg: literal._obj
                                       for arg, literal
                                       in apply_node.named_args}
                distribution = (distribution_name,
                                distribution_args,
                                distribution_kwargs)
                res[name] = distribution
        return x

    walk_utils.walk(space, walk_fn)
    return res


def mongo_url(host="localhost", port=27017, db_name="hyperopt"):
    """
    returns a url for MongoJobs or MongoTrials
    """
    return "mongo://{host}:{port}/{db_name}/jobs".format(
        host=host,
        port=port,
        db_name=db_name,
    )


def mongo_worker(mongo_url):
    # TODO factor out
    exp_key = "exp1"
    workdir = "/tmp"
    poll_interval = 0.1
    reserve_timeout = None  # None = don't timeout
    mj = hyperopt.mongoexp.MongoJobs.new_from_connection_str(mongo_url)

    mworker = hyperopt.mongoexp.MongoWorker(mj,
                                            poll_interval,
                                            workdir=workdir,
                                            logfilename=None,
                                            exp_key=exp_key)
    utils.info("Beginning to run hyperopt mongo worker on pid: %s",
               os.getpid())
    while True:
        # this runs a trial exactly once
        mworker.run_one(reserve_timeout=reserve_timeout,
                        erase_created_workdir=True)
        utils.info("Waiting for hyperopt task")


def html_hyperopt_report(trials_df,
                         n_feature_importance=20,
                         n_partial_dependence1=10,
                         n_partial_dependence2=5,
                         n_feature_summary=10,
                         summary_bins=10,
                         categorical_threshold=10,
                         feature_importance_clf_kwargs=None,
                         decision_tree_percentiles=(10, 20),
                         decision_tree_regressions=("cost", "rank"),
                         decision_tree_depth=4):
    """
    trials_df:
    data frame with key "loss" signifying what to minimize

    n_partial_dependence1:
    number of single feature partial dependence features to use

    n_partial_dependence2:
    number of double feature partial dependence features to use
    NOTE: results in choose(n, 2) plots, so don't make this too large

    categorical_threshold:
    maximum number of unique values for a feature to count as a categorical
    one
    """
    # TODO we might eventually want to take in the space as well, so we
    # know whether or not to log scale a certain variable

    from cottonmouth.html import render
    from sklearn import tree
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble.partial_dependence import plot_partial_dependence
    from matplotlib import pyplot as plt
    import pydot

    def float_format(x):
        return "%.4g" % x

    def table(rows, title=None):
        res = ["div"]
        if title is not None:
            res.append(["h1", title])
        table = ["table.table.table-striped"]
        tbody = ["tbody"]
        for row in rows:
            tr = ["tr"] + [["td", elem] for elem in row]
            tbody.append(tr)
        table.append(tbody)
        res.append(table)
        return res

    def png_to_uri(png_buf):
        b64_encoded = urllib.quote(base64.b64encode(png_buf))
        uri = 'data:image/png;base64,' + b64_encoded
        return uri

    def matplotlib_to_uri(fig=None):
        """
        convert matplotlib figure to b64 encoded uri
        http://stackoverflow.com/questions/5314707/matplot-store-image-in-variable
        """
        if fig is None:
            fig = plt.gcf()

        imgdata = StringIO.StringIO()
        fig.savefig(imgdata, format='png')
        imgdata.seek(0)  # rewind the data

        return png_to_uri(imgdata.buf)

    def tree_to_uri(clf, feature_names):
        """
        converts a decision tree into image as a b64 encoded uri
        """
        dot_data = StringIO.StringIO()
        tree.export_graphviz(clf,
                             out_file=dot_data,
                             # map to str to avoid unicode issues
                             feature_names=map(str, feature_names),
                             # WARNING: these flags require sklearn version >=
                             # 0.17.1
                             filled=True,
                             rounded=True,
                             special_characters=True,
                             )
        graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
        return png_to_uri(graph.create_png())

    # -------------
    # preprocessing
    # -------------

    # prepare X and y for feature importance / partial dependence
    # FIXME handle missing X data (for conditional variables)
    X_df = trials_df.drop(["loss"], axis=1)
    X = X_df.as_matrix()
    # fill in missing y with worst value
    y = np.array(trials_df.loss)
    bad_idxs = np.isnan(y) | np.isinf(y)
    y[bad_idxs] = y[~bad_idxs].max() + 1

    # sort the data
    sorted_trials_df = trials_df.sort_values(["loss"])

    # calculate feature importance
    clf_kwargs = dict(
        n_estimators=500,
        bootstrap=True,
        max_depth=3,
        n_jobs=-1,
        max_features="auto",
        random_state=1
    )
    if feature_importance_clf_kwargs is not None:
        clf_kwargs.update(feature_importance_clf_kwargs)
    clf = ExtraTreesRegressor(**clf_kwargs)
    clf.fit(X, y)
    feature_importances = zip(X_df.columns, clf.feature_importances_)
    feature_importances.sort(key=lambda x: x[1], reverse=True)
    feature_names = [x[0] for x in feature_importances]

    # partial dependence plots
    # FIXME parameterize
    clf_kwargs = dict(
        n_estimators=200,
        max_depth=5,
        subsample=0.5,
        max_features="auto",
        learning_rate=0.1,
        loss='huber',
        random_state=1
    )
    clf = GradientBoostingRegressor(**clf_kwargs)
    clf.fit(X, y)
    X_columns = list(X_df.columns)
    feature_idxs = [X_columns.index(n) for n in feature_names]
    single_feature_idxs = feature_idxs[:n_partial_dependence1]
    double_feature_idxs = feature_idxs[:n_partial_dependence2]
    features = (single_feature_idxs
                + list(itertools.combinations(double_feature_idxs, 2)))

    fig, axs = plot_partial_dependence(clf,
                                       X,
                                       features,
                                       feature_names=X_df.columns,
                                       n_jobs=-1,
                                       grid_resolution=100)
    fig.set_size_inches(9, len(features))
    # leave space for the previous row's x-label
    plt.subplots_adjust(top=0.9)
    pd_plot_uri = matplotlib_to_uri(fig)

    # creating feature summaries
    feature_summaries = []
    for feat_name in feature_names[:n_feature_summary]:
        if len(trials_df[feat_name].unique()) > categorical_threshold:
            # numerical feature
            series = trials_df[feat_name].as_matrix()
            _bins = np.percentile(series, np.linspace(0, 100, 11))
            labels = []
            for _value in series:
                _upper_idx = (_value >= _bins[:-1]).sum()
                _lower = _bins[_upper_idx - 1]
                _upper = _bins[_upper_idx]
                if _upper_idx == len(_bins) - 1:
                    assert _lower <= _value <= _upper
                    labels.append("%f <= x <= %f" % (_lower, _upper))
                else:
                    assert _lower <= _value < _upper
                    labels.append("%f <= x < %f" % (_lower, _upper))
            groupby_df = trials_df.groupby(labels)
        else:
            # categorical feature
            groupby_df = trials_df.groupby(feat_name)
        loss_median = groupby_df.median().loss
        loss_mean = groupby_df.mean().loss
        loss_std = groupby_df.std().loss
        loss_count = groupby_df.count().loss
        summary_df = pd.DataFrame(
            [loss_median, loss_mean, loss_std, loss_count],
            index=["median", "mean", "std", "count"])
        trials_df.plot(feat_name, "loss", kind="scatter")
        feat_plot_uri = matplotlib_to_uri()
        feature_summaries.append(utils.AttrDict(
            feat_name=feat_name,
            plot_uri=feat_plot_uri,
            summary_df=summary_df,
        ))

    # creating decision tree
    tree_titles_and_imgs = []
    for percentile in decision_tree_percentiles:
        tmp_y = y < np.percentile(y, percentile)
        clf = tree.DecisionTreeClassifier(max_depth=decision_tree_depth)
        clf.fit(X, tmp_y)
        title = "Classification tree for best %f%% of samples" % percentile
        tree_img = tree_to_uri(clf, X_df.columns)
        tree_titles_and_imgs.append((title, tree_img))
    for _regression_task in decision_tree_regressions:
        if _regression_task == "cost":
            tmp_y = y
        elif _regression_task == "rank":
            _rank = numpy_utils.to_ranking(y)
            # normalize to [0,1] to be less sensitive to mse
            tmp_y = _rank.astype(float) / _rank.max()
        else:
            raise NotImplementedError("Unknown regression task: %s"
                                      % _regression_task)
        for criterion in ["mse", "mae"]:
            clf = tree.DecisionTreeRegressor(criterion=criterion,
                                             max_depth=decision_tree_depth)
            clf.fit(X, tmp_y)
            title = "Regression (%s) tree for loss: %s" % (criterion,
                                                           _regression_task)
            tree_img = tree_to_uri(clf, X_df.columns)
            tree_titles_and_imgs.append((title, tree_img))

    # ---------------
    # creating report
    # ---------------
    body = ["body"]

    # add in data
    body.append(["div#raw_data",
                 ["h1", "top data points"],
                 # order
                 sorted_trials_df[["loss"] + feature_names].to_html(
                     float_format=float_format),
                 # setting no order to use same order as pandas df
                 ["script", """
                 $('#raw_data table').DataTable({'order': []})
                 """], ])

    # add feature importances
    body.append(table([(n, float_format(v))
                       for n, v in feature_importances[:n_feature_importance]],
                      "feature importance"))

    # partial dependenct plot
    body.append(["div",
                 ["h1", "partial dependence"],
                 ["img", {"src": pd_plot_uri}]])

    # add feature summaries
    feature_summaries_element = ["div", ["h1", "feature summaries"]]
    for summary in feature_summaries:
        elem = ["div"]
        elem.append(["h2", summary.feat_name])
        elem.append(["img", {"src": summary.plot_uri}])
        elem.append(summary.summary_df.to_html(
            float_format=float_format),)
        feature_summaries_element.append(elem)
    body.append(feature_summaries_element)

    # add tree imgs
    tree_imgs_element = ["div", ["h1", "tree images"]]
    for title, tree_img in tree_titles_and_imgs:
        elem = ["div"]
        elem.append(["h2", title])
        elem.append(["img", {"src": tree_img, "style": "width:100%;"}])
        tree_imgs_element.append(elem)
    body.append(tree_imgs_element)

    return render([
        "html",
        ["head",
         ["title", "Hyperopt Report"],
         ["link", {"rel": "stylesheet",
                   "href": ("https://maxcdn.bootstrapcdn.com/bootstrap/"
                            "3.3.1/css/bootstrap.min.css")}],
         ["link", {"rel": "stylesheet",
                   "href": ("https://cdn.datatables.net/1.10.7/css/"
                            "jquery.dataTables.min.css")}],
         ["script", {"src": "https://code.jquery.com/jquery-2.1.4.js"}],
         ["script", {"src": ("https://cdn.datatables.net/1.10.7/js/"
                             "jquery.dataTables.min.js")}]],
        body
    ])
