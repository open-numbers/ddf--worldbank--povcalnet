# encoding: utf-8

"""Smoothing the CDF and get povcalnet shapes

The povcalnet CDFs are noisy. We observed that the tails, where headcount < 10%
and headcount > 90% ard very noisy, so we use a lower sample rate for these groups.
And interpolate values for the missing groups.
Then the CDF is furture smoothed with moving averages.

Then we convert CDFs to income mountain shapes, and smooth it again.

NOTE: There are some shapes with negative values, we remove
those values and interpolate values before smoothing
"""

# %%
import os
import sys

import numpy as np
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import warnings
import psutil
from multiprocessing import get_context
import smoothlib

from scipy.interpolate import PchipInterpolator


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


# method to interpolate the series
def interpolate(df, extrapolate=False):
    ser = df.to_pandas()
    ser_ = ser.dropna()
    res = PchipInterpolator(ser_.index, ser_.values, extrapolate=extrapolate)(ser.index)
    return pl.Series('headcount', res)


# method to resample the data
# we want dense data in the middle, but sparse data in the tails
def resample(df):
    fst = df.filter(
        pl.col('headcount') > 0.1
    ).select(pl.col('i').first()).item()
    lst = df.filter(
        pl.col('headcount') > 0.9
    ).select(pl.col('i').first()).item()

    select_groups = list(range(fst, 0, -10)) + \
        list(range(fst+1, lst)) + \
        list(range(lst, 400, 10)) + \
        [400, 460, 500]
    if select_groups[0] != 0:
        select_groups.insert(0, 0)

    return df.filter(
        pl.col('i').is_in(select_groups)
    ).group_by('headcount').agg(
        pl.col('i').median(),
        pl.col('headcount').last().alias('headcount2')
    ).select(
        pl.col('i').cast(pl.Int32),
        pl.col('headcount2').alias('headcount')
    ).sort('i')


# Step 1.
def step1(res0: pl.DataFrame):
    """remove and fill values

    1. fix diff = 0 by removing all equal values
    2. add back missing groups by interpolation (from 0 to 500)
    """
    all_i = pl.Series('i', range(501), dtype=pl.Int32).to_frame()

    def group_func(df):
        country = df['country'][0]
        year = df['year'][0]
        reporting_level = df['reporting_level'][0]

        df2 = resample(df).join(all_i, on='i', how='full', coalesce=True)
        df3 = df2.with_columns(
            pl.col('headcount').map_batches(interpolate)
            .fill_nan(None)
            .forward_fill().backward_fill()
        )
        return df3.with_columns(
            pl.lit(country).alias('country'),
            pl.lit(year).alias('year'),
            pl.lit(reporting_level).alias('reporting_level')
        )

    return res0.group_by(
        'country', 'year', 'reporting_level').map_groups(group_func)


# res1 = step1(res0)
# df = _f(res0, country='USA', year=2019, reporting_level='national')
# df_ = _f(res1, country='USA', year=2019, reporting_level='national')
# plot(df, diff=True)
# plot(df_, diff=True)
# plt.show()


# Step 2. Smoothing CDF
def smooth_cdf(expr: pl.Expr):

    forward = expr.ewm_mean(span=5, min_periods=1)
    backward = expr.reverse().ewm_mean(span=5, min_periods=1).reverse()

    return (
        (forward + backward) / 2
    )


def step2(res1):
    """Smooth the CDF
    """
    res2 = res1.with_columns(
        smooth_cdf(pl.col('headcount'))
        .over(['country', 'year', 'reporting_level'])
    )
    return res2.with_columns(
        pl.when(pl.col('headcount') < 1e-12)
        .then(pl.lit(0.0))
        .when(pl.col('headcount') > (1 - 1e-12))
        .then(pl.lit(1.0))
        .otherwise(pl.col('headcount'))
        .alias('headcount')
    )

# res2 = step2(res1)
# res2.filter(pl.col('headcount') > 0).sort('headcount')
# df1 = _f(res0, country='CHN', year=1990, reporting_level='national')
# df = _f(res2, country='CHN', year=1990, reporting_level='national')
# df['headcount'][380:460].to_numpy()
# df1
# plot(df1[380:460], diff=False)
# plot(df[380:460], diff=False)
# plt.show()


# Step 3. convert to shapes and fix negative values
def get_reset_loc(df: pl.DataFrame) -> pl.Expr:
    xs = df.filter(
        pl.col("headcount") < 0
    ).select(
        pl.concat_list([pl.col('i'),
                        pl.col('i') + 1,
                        pl.col('i') - 1])
        .reshape((-1, )).unique()
    ).filter(
        pl.col('i').is_between(0, 499)  # 499 is the max group
    )
    return xs.to_series()


def fix_negative(df):
    res = df.with_columns(
        pl.when(pl.col("i").is_in(get_reset_loc(df)))
        .then(pl.lit(None))
        .otherwise(pl.col("headcount")).alias('headcount_new')
    )
    res = res.interpolate()  # shall we use pchip?
    res = res.fill_null(0)  # after interpolate, fill 0 in nulls: they should be in the tail.
    return res


def step3(res2):
    """Convert to shapes, fix negative values
    """
    res3_1 = res2.with_columns([
            pl.col('headcount').diff()
            .over(['country', 'year', 'reporting_level'])
        ]).drop_nulls().with_columns(pl.col('i') - 1)

    return res3_1.group_by(
            'country', 'year', 'reporting_level'
        ).map_groups(fix_negative).select([
            pl.col(['country', 'year', 'reporting_level']),
            pl.col('i').alias('bracket'),
            pl.col('headcount_new').alias('headcount')
        ])


# res3 = step3(res2)
# df1 = _f(res1, country='CHN', year=1990, reporting_level='national')
# df = _f(res3, country='CHN', year=1990, reporting_level='national')
# plot(df1, diff=True)
# plt.plot(df['bracket'], df['headcount'])
# plt.show()


# %%
def func(x):
    """function to smooth a series"""
    run_smooth = smoothlib.run_smooth
    # run smoothing, based on standard deviation
    std = x.std()
    if std < 0.004:
        res = run_smooth(x, 30, 7)
        res = run_smooth(res, 30, 3)
        res = run_smooth(res, 20, 0)
        res = run_smooth(res, 10, 0)
        res = run_smooth(res, 10, 0)
    elif std <= 0.0045 and std > 0.004:
        res = run_smooth(x, 30, 5)
        res = run_smooth(res, 30, 2)
        res = run_smooth(res, 20, 0)
        res = run_smooth(res, 10, 0)
        res = run_smooth(res, 10, 0)
    elif std <= 0.0049 and std > 0.0045:
        res = run_smooth(x, 30, 3)
        res = run_smooth(res, 20, 2)
        res = run_smooth(res, 20, 0)
        res = run_smooth(res, 10, 0)
    elif std > 0.0049:
        res = run_smooth(x, 30, 2)
        res = run_smooth(res, 20, 1)
        res = run_smooth(res, 20, 0)
    # also, make sure it will sum up to 100%
    if res.min() < 0:
        res = res - res.min()
    res = res / res.sum()
    return pl.Series(res)


# %%
# geo, t = 'IND', 2013
# df0 = _f(res1, country=geo, year=t, reporting_level='national')
# df3 = _f(res3, country=geo, year=t, reporting_level='national')
# print(df3['headcount'].std())
# df0_ = df0.with_columns(
#     pl.col('headcount').diff()
# ).drop_nulls().with_columns(pl.col('i') - 1)
# # %%
# # df.select('headcount2').to_series().eval(func)
# df3_ = df3.with_columns(
#     pl.col('headcount').map(func)
# )
# # -> 360 ms
# # %% The time needed
# # res5.select(['country', 'year', 'reporting_level']).unique().shape[0] * 400 / 1000 / 6 / 60
# # %%
# plt.plot(df0_['i'], df0_['headcount'])
# plt.plot(df3_.select('headcount'))
# plt.show()

def smooth_shapes(df):
    return df.with_columns(
        pl.col('headcount').map_batches(func)
    )


# rename things
def step5(res4: pl.DataFrame):
    mapping = {'national': 'n', 'rural': 'r', 'urban': 'u'}

    # MAYBE: change headcount -> population_percentage?
    return res4.with_columns(
        # xkx in povcalnet is kos in gapminder
        pl.col('country').str.to_lowercase().str.replace("xkx", "kos"),
        pl.col('reporting_level').replace_strict(mapping)
    )


# %%
if __name__ == '__main__':
    res0 = pl.read_parquet('./povcalnet_clean.parquet')
    res1 = step1(res0)
    res2 = step2(res1)
    res3 = step3(res2)
    print("check if there are nulls:")
    print(res3.filter(pl.col('headcount').is_null()))

    # have to use multiprocess here. set the pool size
    poolsize = psutil.cpu_count(logical=False) - 1

    with warnings.catch_warnings(record=False) as w:
        with get_context("spawn").Pool(poolsize) as pool:
            todos = res3.partition_by(['country', 'year', 'reporting_level'])
            print(len(todos))
            res4_lst = pool.map(smooth_shapes, todos)

    res4 = pl.concat(res4_lst)

    # don't keep those < 0
    res4 = res4.filter(pl.col('headcount') > 1e-12)

    # print(res4)
    res5 = step5(res4)
    res5.write_parquet('./povcalnet_smoothed.parquet')

    # export all avaliable country/year
    povcal_country_year = res5.select(['country', 'year']).unique()
    povcal_country_year.write_csv('povcal_country_year.csv')

    # TODO: add some more checking images
    plt.figure()
    plt.plot(_f(res0, country='USA', year=2016, reporting_level='national')
             .select('headcount').to_series().diff().drop_nulls(), alpha=.4)
    df = _f(res5, country='usa', year=2016, reporting_level='n')
    plt.plot(df['bracket'], df['headcount'])
    plt.savefig("compare_smoothed.jpg")
    print('check compare_smoothed.jpg for how well the smoothing goes')
