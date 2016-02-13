import os
import subprocess


def memory_usage_ps():
    """
    returns memory consumption for the current process in megabytes

    from: http://fa.bianp.net/blog/2013/different-ways-to-get-memory-consumption-or-lessons-learned-from-memory_profiler/
    """
    out = subprocess.Popen(
        ['ps', 'v', '-p', str(os.getpid())],
        stdout=subprocess.PIPE
    ).communicate()[0].split(b'\n')
    vsz_index = out[0].split().index(b'RSS')
    mem = float(out[1].split()[vsz_index]) / 1024
    return mem
