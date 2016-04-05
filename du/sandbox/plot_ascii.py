# coding=utf-8
"""
modified from https://github.com/shawntan/theano_toolkit/
"""
import numpy as np
import du


def plot_ascii(arr, scale="auto", abs_val=True, epsilon=1e-8, **printoptions):
    chars = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    arr = arr.astype(np.float)
    if scale == "auto":
        if abs_val:
            min_val = 0
            max_val = np.abs(arr).max()
        else:
            min_val = arr.min()
            max_val = arr.max()
    else:
        min_val, max_val = scale

    def visual(val):
        if abs_val:
            val = abs(val)
        step = (val - min_val) / (max_val - min_val + epsilon)
        step *= len(chars)
        step = int(step)
        step = np.clip(step, 0, len(chars) - 1)
        if abs_val and val < 0:
            # change color of bars if we take the abs of negative values
            color_start, color_end = '\033[90m', '\033[0m'
        else:
            color_start, color_end = "", ""
        return color_start + chars[step] + color_end

    with du.numpy_utils.printoptions(**printoptions):
        print np.array2string(
            arr,
            formatter={'float_kind': lambda x: visual(x)},
            max_line_width=10000
        )

if __name__ == "__main__":
    import scipy.misc
    lena = scipy.misc.lena()
    lena_x8 = lena[::8, ::8] + 0.0
    plot_ascii(lena_x8, threshold=100 * 100)
    lena_x8 -= lena_x8.mean()
    lena_x8 /= lena_x8.std()
    plot_ascii(lena_x8, threshold=100 * 100)
