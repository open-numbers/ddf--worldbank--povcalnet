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
from smoothlib import run_smooth

from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.optimize import minimize


# %%
# settings for display images
sns.set_context('notebook')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (7, 4)
plt.rcParams['figure.dpi'] = 144


# %%
def _f(df, **kwargs):
    return df.filter(pl.all_horizontal([(pl.col(k) == v) for k, v in kwargs.items()]))


def plot(df, diff=False):
    if diff:
        plt.plot(df['i']-1, df['headcount'].diff())
    else:
        plt.plot(df['i'], df['headcount'])


# functions for smoothing the CDF.
def preprocess_data(x, y):
    # Record min and max values
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    # Remove consecutive points with the same value
    diff = np.diff(y, prepend=0)
    mask = (np.abs(diff) > 1e-4) & (diff > 0)
    x, y = x[mask], y[mask]

    # Fix the initial part of the CDF
    # We believe that the headcount must be very small at the beginning
    if np.min(y[:200]) > 0.001:
        y200 = y[200]
        # Remove the first 200 points
        x = x[200:]
        y = y[200:]
        # Append two new points: (0, 0) and (100, y[200] / 2)
        x = np.insert(x, 0, 0)
        y = np.insert(y, 0, 0)
        x = np.insert(x, 1, 100)
        y = np.insert(y, 1, y200 / 2)

    # Append or set min_x to min_y and max_x to max_y
    if x[0] != min_x:
        x = np.insert(x, 0, min_x)
        y = np.insert(y, 0, min_y)
    else:
        y[0] = min_y

    if x[-1] != max_x:
        x = np.append(x, max_x)
        y = np.append(y, max_y)
    else:
        y[-1] = max_y

    # also extend right side
    x = np.append(x, 500)
    y = np.append(y, 1)

    return x, y


def estimate_noise_level(y):
    # Estimate noise using the difference between adjacent points
    diff = np.diff(y)
    noise_level = np.std(diff)
    return noise_level


def adaptive_window_length(noise_level, n_points):
    # Adjust window length based on noise level and number of points
    # Start with 10% of data points, minimum 3
    base_window = max(int(n_points * 0.1), 3)
    noise_factor = int(noise_level * 2000)  # Scale noise to usable range

    window = base_window + noise_factor
    # Ensure window is smaller than data
    window = min(window, (n_points - 1) // 2)
    window = window if window % 2 == 1 else window + 1  # Ensure odd number

    return window


def smooth_and_monotonize_cdf(x, y):
    # Preprocess the data
    x_processed, y_processed = preprocess_data(x, y)

    noise_level = estimate_noise_level(y_processed)
    n_points = len(x_processed)
    window_length = adaptive_window_length(noise_level, n_points)
    polyorder = min(3, window_length - 1)  # Adjust polyorder if necessary

    # Step 1: Smooth the data using Savitzky-Golay filter
    y_smoothed = savgol_filter(y_processed, window_length, polyorder)

    # Step 2: Ensure monotonicity
    y_monotone = np.maximum.accumulate(y_smoothed)

    # Step 3: Clip values to [0, 1] range
    y_monotone = np.clip(y_monotone, 0, 1)

    # Step 3: Create a monotonic interpolation
    # f = interpolate.PchipInterpolator(x_processed, y_monotone)
    f = interpolate.interp1d(
        x_processed, y_monotone, kind='linear', bounds_error=False, fill_value=(0, 1))

    return f


# function to smooth the PDF.
def smooth_pdf(x, y, smoothness=1.0, max_iterations=100, constraint_interval=5):
    """
    Smooth the input PDF while preserving a single maximum at the midpoint of observed maxima,
    and ensuring strict monotonicity before and after the maximum with reduced constraints.
    """
    # Find all indices of the maximum value
    max_value = np.max(y)
    max_indices = np.where(y == max_value)[0]
    area = np.sum(y)

    # Calculate the midpoint index of the maximum values
    mid_max_index = int(np.mean(max_indices))

    # Define the objective function to minimize
    def objective(y_smooth):
        # Smoothness term
        smoothness_term = np.mean(np.diff(y_smooth, 2)**2)
        # Fit term
        fit_term = np.mean((y - y_smooth)**2)
        return smoothness * smoothness_term + fit_term

    # Define constraints
    constraints = [
        # Ensure the total area under the curve remains the same
        {'type': 'eq', 'fun': lambda y_smooth: np.sum(y_smooth) - area},
        # Preserve the maximum value at the midpoint
        {'type': 'eq',
            'fun': lambda y_smooth: y_smooth[mid_max_index] - max_value},

        # Ensure all y values are non-negative
        {'type': 'ineq', 'fun': lambda y_smooth: y_smooth}]

    # Add reduced number of monotonicity constraints
    for i in range(0, mid_max_index, constraint_interval):
        constraints.append(
            {'type': 'ineq', 'fun': lambda y_smooth,
             i=i: y_smooth[i+constraint_interval] - y_smooth[i] - 1e-10}
        )

    for i in range(mid_max_index, len(y) - constraint_interval, constraint_interval):
        constraints.append(
            {'type': 'ineq', 'fun': lambda y_smooth,
             i=i: y_smooth[i] - y_smooth[i+constraint_interval] - 1e-10}
        )

    # Minimize the objective function
    result = minimize(objective, y, method='SLSQP',
                      constraints=constraints, options={'maxiter': max_iterations})

    return result.x


# This function also smooth the shape, based on averaging on a dynamic window
# but it will change the shape slightly so we only apply this after applying the function above.
def func(x):
    """function to smooth a series"""
    # run smoothing, based on standard deviation
    std = x.std()
    s0 = np.sum(x)
    if std < 0.004:
        res = run_smooth(x, 30, 7)
        res = run_smooth(res, 30, 3)
    elif std <= 0.0045 and std > 0.004:
        res = run_smooth(x, 30, 5)
        res = run_smooth(res, 30, 2)
    elif std <= 0.0049 and std > 0.0045:
        res = run_smooth(x, 30, 3)
        res = run_smooth(res, 20, 2)
    elif std > 0.0049:
        res = run_smooth(x, 30, 2)
        res = run_smooth(res, 20, 1)
    s1 = np.sum(res)
    correction_factor = s0 / s1
    return correction_factor * res


def create_smoothed_shape(df):
    xs = df['i']
    ys = df['headcount']
    smoothed_cdf = smooth_and_monotonize_cdf(xs.to_numpy(), ys.to_numpy())

    xs_ = np.arange(0, 501, 1)
    smoothed_ys = smoothed_cdf(xs_)

    shape_xs = xs_[:-1]  # 0 - 500
    shape_ys = np.diff(smoothed_ys)

    smoothed_y = smooth_pdf(
        shape_xs, shape_ys, smoothness=50, max_iterations=5, constraint_interval=5)

    smoothed_y = func(smoothed_y)

    country = df['country'][0]
    year = df['year'][0]
    level = df['reporting_level'][0]    
    return pl.DataFrame({'i': shape_xs, 'headcount': smoothed_y}).select(
        pl.lit(country).alias('country'),
        pl.lit(year).alias('year'),
        pl.lit(level).alias('reporting_level'),
        pl.col(['i', 'headcount'])
    )
    

# rename things
def rename_things(res1: pl.DataFrame):
    mapping = {'national': 'n', 'rural': 'r', 'urban': 'u'}

    # MAYBE: change headcount -> population_percentage?
    return res1.with_columns(
        # xkx in povcalnet is kos in gapminder
        pl.col('country').str.to_lowercase().str.replace("xkx", "kos"),
        pl.col('reporting_level').replace_strict(mapping)
    ).rename({'i': 'bracket'})


# %%
if __name__ == '__main__':
    res0 = pl.read_parquet('./povcalnet_clean.parquet')
    # have to use multiprocess here. set the pool size
    poolsize = psutil.cpu_count(logical=True) - 2

    with warnings.catch_warnings(record=False) as w:
        with get_context("spawn").Pool(poolsize) as pool:
            todos = res0.partition_by(['country', 'year', 'reporting_level'])
            print(len(todos))
            res1_lst = pool.map(create_smoothed_shape, todos)

    res1 = pl.concat(res1_lst)

    # don't keep those with very low headcount
    res1 = res1.filter(pl.col('headcount') > 1e-13)

    # print(res4)
    res = rename_things(res1)
    res.write_parquet('./povcalnet_smoothed.parquet')

    # export all avaliable country/year
    povcal_country_year = res.select(['country', 'year']).unique()
    povcal_country_year.write_csv('povcal_country_year.csv')

    # TODO: add some more checking images
    plt.figure()
    plt.plot(_f(res0, country='USA', year=2016, reporting_level='national')
             .select('headcount').to_series().diff().drop_nulls(), alpha=.4)
    df = _f(res, country='usa', year=2016, reporting_level='n')
    plt.plot(df['bracket'], df['headcount'])
    plt.savefig("compare_smoothed.jpg")
    print('check compare_smoothed.jpg for how well the smoothing goes')
