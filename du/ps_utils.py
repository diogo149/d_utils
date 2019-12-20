import os
import psutil
import time


def process_time():
    p = psutil.Process(os.getpid())
    return time.time() - p.create_time()
