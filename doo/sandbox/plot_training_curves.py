from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import matplotlib
import matplotlib.pyplot as plt
try:
    from .. import utils
except ValueError:
    # if using as a standalone script
    from d import utils


def plot_training_curves(
        data,
        title=None,
        filename=None,
        # markers=["-", "--", "o--", "o-"],
        markers=matplotlib.markers.MarkerStyle.filled_markers,
        data_cmap=matplotlib.cm.rainbow,
        # data_cmap=matplotlib.cm.Set1,
        y_styles=["-", "--", "_", ":", ""],
        y_cmap=matplotlib.cm.rainbow):
    """
    data:
    list of dicts with the following keys:
    - name: eg. "train loss"
      - corresponds to marker in plot
    - x: sequence of points eg. time, or epoch number
    - x_unit: string, description of unit of x (eg. "time (s)" or "epoch num")
      - all dicts must have the same x_unit
    - y: sequence of points eg. loss or AUC
    - y_unit: string, description of unit of y (eg. "loss" or "AUC")
    - data_id
      - correspond to color in plot

    filename:
    to save to a file instead of showing

    heavily based on:
    http://matplotlib.org/examples/api/two_scales.html
    """
    # must be only 1
    x_unit, = set(m["x_unit"] for m in data)

    y_units = list(utils.toolz.unique(m["y_unit"] for m in data))
    if len(y_units) >= 3:
        utils.warn_once("Plotting curves with more than 2 y_unit's")
    # TODO generic sequence to color given sequence and color map
    if len(y_units) > 1:
        y_unit_to_color = {y: y_cmap(idx / (len(y_units) - 1))
                           for idx, y in enumerate(y_units)}
    else:
        y_unit_to_color = {y_units[0]: y_cmap(0.5)}

    data_ids = list(utils.toolz.unique(m["data_id"] for m in data))
    if len(data_ids) > 1:
        data_id_to_color = {n: data_cmap(idx / (len(data_ids) - 1))
                            for idx, n in enumerate(data_ids)}
    else:
        data_id_to_color = {data_ids[0]: data_cmap(0.5)}

    names = list(utils.toolz.unique(m["name"] for m in data))
    assert len(names) <= len(markers)

    # create axes
    fig = plt.figure()
    # leave space on the right for legend
    orig_ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
    axes = [orig_ax]
    while len(axes) < len(y_units):
        axes.append(axes[-1].twinx())

    # plot data
    lines = []
    line_names = []
    for m in data:
        ax = axes[y_units.index(m["y_unit"])]
        color = data_id_to_color[m["data_id"]]
        name_idx = names.index(m["name"])
        marker = markers[name_idx]
        style = y_styles[y_units.index(m["y_unit"])]
        line, = ax.plot(m["x"],
                        m["y"],
                        marker + style,
                        color=color,
                        # don't fill markers
                        markerfacecolor='none',
                        # add marker color back (from markerfacecolor="none")
                        markeredgecolor=color)
        lines.append(line)
        line_names.append("{}:{}".format(m["data_id"], m["name"]))

    # set colors of y axes
    for idx, ax in enumerate(axes):
        y_color = y_unit_to_color[y_units[idx]]
        ax.set_ylabel(y_units[idx], color=y_color)
        for ticklabel in ax.get_yticklabels():
            ticklabel.set_color(y_color)

    if title is not None:
        orig_ax.set_title(title)

    orig_ax.set_xlabel(x_unit)

    # make font size somehwat small, just in case some line_names are very
    # long
    fig.legend(lines, line_names, "right", prop={'size': 8})

    if filename is None:
        fig.show()
    else:
        fig.savefig(filename)


if __name__ == "__main__":
    import numpy as np
    t = np.arange(0.01, 10.0, 0.01)
    s1 = np.exp(t)
    s2 = np.sin(2 * np.pi * t)

    plot_training_curves([
        dict(name="foo", x=t, y=s1, x_unit="t", y_unit="exp", data_id=0),
        dict(name="foo2", x=t, y=s2, x_unit="t", y_unit="sin", data_id=0),
        dict(name="foo", x=t, y=s1 * 1.5, x_unit="t", y_unit="exp",
             data_id=1),
    ],
        title="this is a test",
        filename="foo.png",
    )
