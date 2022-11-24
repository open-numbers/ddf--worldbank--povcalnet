import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def mval_at_point(x, y, x_y_max, y_max, maximum=10, minimum=0):
    """get the half of window size (M value) of given point.

    Based on x and y distance to the peak position.
    """
    x_d = (abs(x - x_y_max) / x_y_max)**2
    y_d = (abs(y - y_max) / y_max)**2

    if y_d == 0:
        return minimum

    d = maximum * (0.8 * x_d + 0.2 * y_d)

    d = int(d)

    if d <= minimum:
        return minimum
    if d > maximum:
        return maximum
    else:
        return d


def get_sample_at_point(x, arr, y_max, x_y_max, mval=None):
    y = arr[x]
    # y_max = arr.max()
    # x_y_max = np.where(arr == y_max)[0][0]
    if not mval:
        mval = mval_at_point(x, y, x_y_max, y_max)

    if x + mval + 1 > len(arr):
        pad = x + mval - len(arr) + 1
        ending = np.zeros(pad)
    else:
        ending = np.array([])

    if x < mval:
        pad = mval - x
        beginning = np.zeros(pad)
    else:
        beginning = np.array([])

    res = np.concatenate([beginning, arr, ending])

    if len(beginning) > 0:
        return res[:(mval * 2 + 1)]
    else:
        return res[(x - mval):(x + mval + 1)]


def tricubic(x):
    y = np.ones_like(x)
    # idx = (x >= -1) & (x <= 1)
    max_d = x.max() - x.min()
    xmin = x.min()
    if max_d == 0:
        return y
    for idx, xv in enumerate(x):
        y[idx] = np.power(1.0 - np.power(np.abs(x[idx] - xmin), 3), 3)
    return y


def func(x, a, b):
    return a * x + b


def estimate(sample, weights, xpos):
    x = list(range(len(sample)))
    y = sample
    popt, pcov = curve_fit(func, x, y, sigma=weights)
    # print(popt)
    return popt[0] * xpos + popt[1]


def run_smooth(arr, maximum=10, minimum=1):
    """smooth a curve
    
    maximum: controls the smoothness of the tails
    minimum: controls the smoothness near the peak
    """
    res = []
    y_max = arr.max()
    x_y_max = np.where(arr == arr.max())[0][0]
    for i in range(len(arr)):
        # print(i, end=',')
        y = arr[i]
        #     if x_y_max == i:
        #         res.append(ser.max())
        #         print()
        #         continue
        mval = mval_at_point(i,
                             y,
                             x_y_max,
                             y_max,
                             maximum=maximum,
                             minimum=minimum)
        # print(mval)
        if mval == 0:
            res.append(y)
            continue
        sample = get_sample_at_point(i, arr, y_max, x_y_max, mval=mval)
        # print(sample)
        # assert len(
        #     sample) == 2 * mval + 1, f"i={i}, size={len(sample)}, mval={mval}"
        weights = tricubic(sample)
        res.append(estimate(sample, weights, mval))
    res = pd.Series(res)
    res[res < 10**(-20)] = 0  # remove if numbers are too small
    return res
