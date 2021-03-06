"""
based on:
https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error

simply import:
import du.sandbox.auto_pdb_on_error
"""

import sys


def info(type, value, tb):
    if hasattr(sys, "ps1") or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback
        import ipdb

        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        # TODO should we use ipdb instead
        ipdb.post_mortem(tb)  # more "modern"


sys.excepthook = info
