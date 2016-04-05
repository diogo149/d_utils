from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import pylab
import matplotlib
import du

from .plot_training_curves import plot_training_curves


# TODO factor out pylab.show() into a global setting to flip between
# saving a figure, showing, or returning the image
# or figure out how to return just the image


def plot_patches_in_grid(patches,
                         size_in_inches=(12, 12),
                         max_cols=6,
                         cmap=None):
    """
    plots serveral 2D patches in a square-ish grid
    """
    n_cols = min(max_cols, int(np.sqrt(len(patches))))
    n_rows = int(np.ceil(len(patches) / n_cols))
    assert n_rows * n_cols >= len(patches)
    for i, s in enumerate(patches):
        pylab.subplot(n_rows, n_cols, i + 1)
        pylab.imshow(s, cmap=cmap)
        pylab.axis("off")
    fig = pylab.gcf()
    # TODO make this a global setting as well?
    fig.set_size_inches(*size_in_inches)
    pylab.show()


def plot_cmap(cmap):
    """
    plot a gradient of a color map
    """
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    pylab.imshow(gradient, aspect='auto', cmap=cmap)
    pylab.show()


def plot_path(path):
    fig, ax = pylab.subplots()
    patch = matplotlib.patches.PathPatch(path, facecolor='r', alpha=0.5)
    ax.add_patch(patch)

    # plot control points and connecting lines
    x, y = zip(*path.vertices)
    line, = ax.plot(x, y, 'go-')

    ax.grid()
    ax.axis('equal')
    pylab.show()


# XXX This function is deprecated in favor of the monitor_ui
# def plot_trial_data(trial_name,
#                     iteration_nums,
#                     x_key,
#                     x_unit,
#                     y_pairs,
#                     x_limit=None,
#                     trials_dir=utils.config["trial"]["trials_dir"],
#                     filename=None):
#     """
#     iteration_nums:
#     - which iterations to plot for

#     x_key:
#     - key in store for x data (must be same for all y's)

#     x_unit:
#     - label for x

#     y_pairs:
#     - list of tuples of y_key and y_unit
#     """
#     import numpy as np
#     data = []
#     for iteration_num in iteration_nums:
#         trial = du.trial.TrialState(trial_name, iteration_num, trials_dir)
#         trial.load()
#         for y_key, y_unit in y_pairs:
#             x = np.array(trial.get(x_key))
#             y = np.array(trial.get(y_key))
#             if x_limit is not None:
#                 idx = np.where(x <= x_limit)
#                 x = x[idx]
#                 y = y[idx]
#             data.append(dict(
#                 name=y_key,
#                 x=x,
#                 x_unit=x_unit,
#                 y=y,
#                 y_unit=y_unit,
#                 data_id=iteration_num,
#             ))
#     plot_training_curves(
#         data=data,
#         filename=filename,
#     )
