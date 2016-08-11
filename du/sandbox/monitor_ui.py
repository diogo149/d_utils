"""
UI for monitoring data

features
- easy setup (simply run `python -m SimpleHTTPServer` in the directory)
- live monitoring (before all the data is done)
- customizable with data
  - eg. linear/log scale, rolling mean window
- multiple charts
- tooltips of the highlighted data point
- click the legend to hide lines
- draggable legend
"""

import re
import os
import shutil

import numpy as np

import du


class ResultWriter(object):

    """
    use cases:
    - live monitoring w/ a different process
    """

    def __init__(self,
                 dirname,
                 pattern="",
                 remove_matched=False,
                 symlink=True,
                 default_settings_file=None):
        """
        pattern:
        regex for which keys to match

        remove_matched:
        whether or not to remove the matched results from the output map

        symlink:
        whether or not to make a symlink instead of copying the html/js files
        (pro: this allow updating the files in the original source)
        """
        self.dirname = dirname
        self.pattern = pattern
        self.remove_matched = remove_matched
        self._jsonl_path = os.path.join(self.dirname, "monitor.jsonl")
        self._regex = re.compile(self.pattern)
        if symlink:
            # create directory
            os.mkdir(dirname)
            # symlink javascript and html
            for f in ["index.html", "monitor.js"]:
                os.symlink(du.templates.template_path("monitor_ui", f),
                           os.path.join(dirname, f))
            # create monitor.jsonl file
            du.io_utils.guarantee_exists(self._jsonl_path)
            if default_settings_file is not None:
                os.symlink(os.path.realpath(default_settings_file),
                           os.path.join(dirname, "default_settings.json"))
        else:
            du.templates.copy_template("monitor_ui", dirname)
            if default_settings_file is not None:
                shutil.copy(os.path.realpath(default_settings_file),
                            os.path.join(dirname, "default_settings.json"))

    def write(self, res):
        # prepare data
        monitor_keys = []
        monitor_data = {}
        for key in res:
            if self._regex.match(key):
                monitor_keys.append(key)
                val = res[key]
                # convert numpy arrays into json serializable format
                if isinstance(val, (np.ndarray, np.number)):
                    val = val.tolist()
                monitor_data[key] = val

        # convert to json and write to file
        du.io_utils.jsonl_append(
            monitor_data, self._jsonl_path, allow_nan=False)

        # optionally remove keys (mutating the result)
        if self.remove_matched:
            for key in monitor_keys:
                res.pop(key)
