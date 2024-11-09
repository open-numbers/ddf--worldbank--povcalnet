# encoding: utf-8

"""Smoothing the CDF and get povcalnet shapes

The povcalnet CDFs are noisy. We apply smoothing to CDF, and then also
apply smoothing to the PDF generated from it. The PDFs are the income
mountain shapes.
"""

# %%
import os
import sys

import numpy as np
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import psutil
from multiprocessing import get_context

from scipy.signal import savgol_filter
from functools import partial


# %%
# settings for display images
sns.set_context("notebook")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (7, 4)
plt.rcParams["figure.dpi"] = 144


# helper to filter a polars dataframe
def _f(df, **kwargs):
    return df.filter(pl.all_horizontal([(pl.col(k) == v) for k, v in kwargs.items()]))


#
def variable_savgol_filter(y, window_func, polyorder=2):
    """
    Apply Savitzky-Golay filter with variable window size.

    Parameters:
    -----------
    y : array-like
        Input signal to be filtered
    window_func : callable
        Function that takes index position (0 to 1) and returns window size
        Should return odd integers
    polyorder : int
        Order of polynomial to fit

    Returns:
    --------
    numpy.ndarray
        Filtered signal
    """
    y = np.array(y)
    n_points = len(y)
    y_filtered = np.zeros(n_points)

    # Apply filter for each point with custom window
    for i in range(n_points):
        # Get window size for this position
        window = int(window_func(i))
        if window % 2 == 0:
            window += 1  # Ensure odd window size

        # Handle edge cases where window would extend beyond signal
        half_window = window // 2
        if i < half_window:
            window = 2 * i + 1
        elif i >= n_points - half_window:
            window = 2 * (n_points - i - 1) + 1

        # Apply Savitzky-Golay filter with computed window
        start_idx = max(0, i - window // 2)
        end_idx = min(n_points, i + window // 2 + 1)
        segment = y[start_idx:end_idx]

        if len(segment) >= polyorder + 1:
            filtered_value = savgol_filter(
                segment, len(segment), polyorder, mode="nearest"
            )
            y_filtered[i] = filtered_value[len(segment) // 2]
        else:
            y_filtered[i] = y[i]  # Use original value if window too small

    return y_filtered


def fwhw(y):
    """
    Calculate the full width at half maximum.

    see https://en.wikipedia.org/wiki/Full_width_at_half_maximum

    We use the width for determining the windows size at a position.
    """
    dif = y.max() - y.min()
    hm = dif / 2
    nearest = (np.abs(y - hm)).argmin()
    middle = y.argmax()
    width = np.abs(middle - nearest) * 2
    # print(nearest, middle, width)
    if middle > nearest:
        return nearest, nearest + width
    else:
        return nearest - width, nearest


def window_func(pos, max_window, hw_left, hw_right, total_x=461):
    """
    Calculate the window size based on the full width position
    """
    min_window = int((hw_right - hw_left) / 5)
    if min_window < 3:
        min_window = 3

    if pos < hw_left:  # Exponential decay region
        # Calculate decay constant
        # We want: max_window * exp(-k * 0.3) = min_window
        # Therefore: k = -ln(min_window/max_window) / 0.3
        k = -np.log(min_window / max_window) / hw_left
        window = max_window * np.exp(-k * pos)

    elif pos > hw_right:  # Exponential growth region
        # Similar calculation but for growth from 0.7 to 1.0
        right_remaining = total_x - hw_right
        k = np.log(max_window / min_window) / right_remaining
        window = min_window * np.exp(k * (pos - hw_right))

    else:  # Constant region
        window = min_window

    # Ensure window size is odd
    window = int(round(window))
    if window > max_window:
        window = max_window
    if window % 2 == 0:
        window += 1

    return window


def create_smooth_pdf_shape_(noisy_cdf, max_window=150, polyorder=3, smooth_epochs=20):
    left, right = fwhw(np.diff(noisy_cdf))
    wf = partial(window_func, max_window=max_window, hw_left=left, hw_right=right)

    y = noisy_cdf
    for i in range(smooth_epochs):
        y = variable_savgol_filter(y, wf, polyorder=polyorder)
        y = np.maximum.accumulate(y)
        y = np.clip(y, 0, 1)

    return np.diff(y)


def create_smooth_pdf_shape(df: pl.DataFrame):
    country = df["country"].unique().item()
    year = df["year"].unique().item()
    reporting_level = df["reporting_level"].unique().item()

    return df.select(
        pl.lit(country).alias("country"),
        pl.lit(year).alias("year"),
        pl.lit(reporting_level).alias("reporting_level"),
        pl.arange(0, 460).alias("bracket"),
        pl.col("headcount").map_batches(create_smooth_pdf_shape_),
    )


def plot(df, diff=False):
    if diff:
        plt.plot(df["i"] - 1, df["headcount"].diff())
    else:
        plt.plot(df["i"], df["headcount"])


# rename things
def rename_things(res1: pl.DataFrame):
    mapping = {"national": "n", "rural": "r", "urban": "u"}

    # MAYBE: change headcount -> population_percentage?
    return res1.with_columns(
        # xkx in povcalnet is kos in gapminder
        pl.col("country").str.to_lowercase().str.replace("xkx", "kos"),
        pl.col("reporting_level").replace_strict(mapping),
    )


# %%
if __name__ == "__main__":
    res0 = pl.read_parquet("./povcalnet_clean.parquet")
    # have to use multiprocess here. set the pool size
    poolsize = psutil.cpu_count(logical=True) - 2

    with warnings.catch_warnings(record=False) as w:
        with get_context("spawn").Pool(poolsize) as pool:
            todos = res0.partition_by(["country", "year", "reporting_level"])
            print(len(todos))
            res1_lst = pool.map(create_smooth_pdf_shape, todos)

    res1 = pl.concat(res1_lst)

    # print(res4)
    res = rename_things(res1)
    res.write_parquet("./povcalnet_smoothed.parquet")

    # export all avaliable country/year
    povcal_country_year = res.select(["country", "year"]).unique()
    povcal_country_year.write_csv("povcal_country_year.csv")

    # TODO: add some more checking images
    plt.figure()
    plt.plot(
        _f(res0, country="IND", year=2020, reporting_level="national")
        .select("headcount")
        .to_series()
        .diff()
        .drop_nulls(),
        alpha=0.4,
    )
    df = _f(res, country="ind", year=2020, reporting_level="n")
    plt.plot(df["bracket"], df["headcount"])
    plt.savefig("compare_smoothed.jpg")
    print("check compare_smoothed.jpg for how well the smoothing goes")
