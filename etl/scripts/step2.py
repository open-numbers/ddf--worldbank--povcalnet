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


def create_pdf_and_remove_outliners(df: pl.DataFrame, p: int):
    pdf = df.select(
        pl.col('i').shift(1),
        pl.col('headcount').diff(),
    ).drop_nulls()
    pdf = pdf.to_pandas()
    pdf['mean'] = pdf['headcount'].rolling(3, min_periods=1, center=True).mean()
    m = pdf['headcount'] - pdf['mean']
    mask = m > p * m.std()

    # set the masked to null
    pdf.loc[mask, 'headcount'] = None
    pdf['headcount'] = pdf['headcount'].interpolate('linear', limit_direction='both')
    return pl.DataFrame(pdf[['i', 'headcount']])


def generate_weights(values):
    """
    Generate weights from an array of non-negative values where:
    - Zero values get zero weights
    - Non-zero values get weights proportional to their difference from the maximum value
    - Values to the left of the maximum get 4x of weights because the noise usually happend on the left.
    - All weights sum to 1

    Parameters:
    values (array-like): 1D array of non-negative numbers

    Returns:
    numpy.ndarray: Array of weights that sum to 1
    """
    # Convert input to numpy array and verify non-negative values
    values = np.array(values)
    if np.any(values < 0):
        raise ValueError("All values must be non-negative")

    # Create mask for non-zero values
    non_zero_mask = values > 0

    # Initialize weights array with zeros
    weights = np.zeros_like(values, dtype=float)

    if np.any(non_zero_mask):
        # Get maximum value and its position
        max_position = np.argmax(values)

        # Create position bias multiplier (2 for left side, 1 for right side)
        position_multiplier = np.ones_like(values)
        position_multiplier[:max_position] = 1.2  # more weights for left side

        # Apply position multiplier to differences
        weighted_differences = values * position_multiplier

        # For non-zero values, use the weighted differences
        valid_mask = non_zero_mask
        valid_differences = weighted_differences[valid_mask]

        # If all differences are 0 (all values are equal), use equal weights
        if np.all(valid_differences == 0):
            weights[valid_mask] = 1 / np.sum(valid_mask)
        else:
            # Normalize the differences to get weights
            weights[valid_mask] = valid_differences / np.sum(valid_differences)

    return weights


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


def find_fwhm_range(cdf_values, lower=0.3, upper=0.7):
    # Find indices where CDF is between lower and upper bounds
    mask = (cdf_values >= lower) & (cdf_values <= upper)
    indices = np.where(mask)[0]

    # Get the corresponding CDF values
    values_in_range = cdf_values[mask]

    return indices, values_in_range


def remove_tails(y, a, b):
    y[:a] = 0
    y[b:] = 0
    return y


def fwhm(y, a, b):
    """
    Calculate the full width at half maximum.

    see https://en.wikipedia.org/wiki/Full_width_at_half_maximum

    We use the width for determining the windows size at a position.
    """
    # we assume that the top of the shapes must be near the middle of the shape
    # not in the tails. So we just reset the tails to 0 to calculate fwhw
    y = remove_tails(y, a, b)
    # Calculate half maximum
    peak_height = y.max() - y.min()
    half_max = y.min() + peak_height / 2

    # Find peak position
    middle = y.argmax()

    # Function to check if 10 consecutive points are below half max
    def check_consecutive_points(start_idx, direction=1):
        consecutive_count = 0
        idx = start_idx

        while 0 <= idx < len(y):
            if y[idx] < half_max:
                consecutive_count += 1
                if consecutive_count >= 10:
                    # Return the first point where it went below half max
                    return idx - (4 if direction > 0 else 0)
            else:
                consecutive_count = 0
            idx += direction

        return None

    # Search for left boundary (moving backwards from peak)
    left_boundary = check_consecutive_points(middle, direction=-1)

    # Search for right boundary (moving forwards from peak)
    right_boundary = check_consecutive_points(middle, direction=1)

    # Handle edge cases where boundaries aren't found
    if left_boundary is None:
        left_boundary = 0
    if right_boundary is None:
        right_boundary = len(y) - 1
    # width = right_boundary - left_boundary
    return left_boundary, right_boundary


def window_func(pos, max_window, hw_left, hw_right, total_x=461, min_window=3):
    """
    """
    # the min window based on data
    min_window_ = int((hw_right - hw_left) / 5)
    if min_window_ > min_window:
        min_window = min_window_

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


# fix the beginning. we assume that in the beginning the shape will be monotonly increasing
def fix_head(y, a):
    y[:a] = np.sort(y[:a])
    return y


def create_smooth_pdf_shape_(noisy_cdf):
    # for a normal distubrition, 3.29 sigma captures 99.9% points.
    pdf = create_pdf_and_remove_outliners(noisy_cdf, 3.3)
    pdf_ = pdf.select(
        (1 - pl.col('headcount').sum()) *
        pl.col('headcount').map_batches(
            generate_weights) + pl.col('headcount')
    )
    clean_cdf = pdf_.select(
        pl.lit(0).append(pl.col('literal')).cum_sum().alias('headcount')
    )['headcount'].to_numpy()

    idxs, _ = find_fwhm_range(clean_cdf)
    a = idxs[0]
    b = idxs[-1]

    left, right = fwhm(np.diff(clean_cdf), a, b)

    # first, use polyorder = 1 and smaller window to reduce noise
    wf = partial(window_func, max_window=40, hw_left=left, hw_right=right, 
                 min_window=1)

    y = clean_cdf
    for i in range(2):
        y = variable_savgol_filter(y, wf, polyorder=1)
        y = np.clip(y, 0, 1)

    # then, use polyorder = 2 and bigger window to smooth the shape
    wf = partial(window_func, max_window=150, hw_left=left, hw_right=right,
                 min_window=5)

    for i in range(10):
        y = variable_savgol_filter(y, wf, polyorder=2)
        y = np.clip(y, 0, 1)

    # ensure the monotone
    y = np.maximum.accumulate(y)

    pdf = np.diff(y)

    if np.max(pdf[:a]) > np.max(pdf[a:]):
        # the shape is strange because there is another peak in very poor side.
        return False, pdf
    else:
        return True, fix_head(pdf, a)


def create_smooth_pdf_shape(df: pl.DataFrame):
    country = df["country"].unique().item()
    year = df["year"].unique().item()
    reporting_level = df["reporting_level"].unique().item()
    # cdf = df['headcount'].to_numpy()

    good_shape, pdf = create_smooth_pdf_shape_(df)
    if not good_shape:
        print(f"bad shape detected: {country}, {year}, {reporting_level}")

    return pl.DataFrame({
                            "country": country,
                            'year': year,
                            'reporting_level': reporting_level,
                            'bracket': np.arange(0, 460),
                            'headcount': pdf
                        })


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
    for country, year, reporting_level in [
        ('IND', 2020, 'national'),
        ('SWE', 2024, 'national'),
        ('CHN', 1983, 'national'),
        ('PAN', 1992, 'national'),
        ('USA', 2002, 'national')
    ]:
        plt.figure()
        plt.plot(
            _f(res0, country=country, year=year, reporting_level=reporting_level)
            .select("headcount")
            .to_series()
            .diff()
            .drop_nulls(),
            alpha=0.4,
        )
        df = _f(res, country=country.lower(), year=year, reporting_level=reporting_level[0])
        plt.plot(df["bracket"], df["headcount"])
        plt.savefig(f"compare_smoothed_{country.lower()}.jpg")
    print("check compare_smoothed_*.jpg for how well the smoothing goes")
