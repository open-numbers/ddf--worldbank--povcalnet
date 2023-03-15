# encoding: utf-8

# %%
import os
import sys

import numpy as np
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns
# %%
# settings for display images
sns.set_context('notebook')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (7, 4)
plt.rcParams['figure.dpi'] = 144
# %%
import pickle
import warnings
from multiprocessing import get_context
import smoothlib

# %%
def _f(df, **kwargs):
    return df.filter(pl.all([(pl.col(k) == v) for k, v in kwargs.items()]))

# %%
def step4(res3):
    # smooth twice
    return res3.with_columns(
        pl.col('headcount')
        .ewm_mean(span=5, min_periods=1).shift(-2).forward_fill()
        .ewm_mean(span=3, min_periods=1).shift(-1).forward_fill()
        .over(['country', 'year', 'reporting_level'])
    )
# %%
def get_reset_loc(df: pl.DataFrame) -> pl.Expr:
    xs = df.filter(
        pl.col("headcount") < 0
    ).select(
        pl.concat_list([pl.col('i'),
                        pl.col('i') + 1,
                        pl.col('i') - 1])
        .reshape((-1, )).unique()
    ).filter(
        pl.col('i').is_between(0, 459)
    )
    return xs.to_series()

def fix_negative(df):
    res = df.with_columns(
        pl.when(pl.col("i").is_in(get_reset_loc(df)))
        .then(pl.lit(None))
        .otherwise(pl.col("headcount")).alias('headcount_new')
    )
    return res.interpolate()

def expand_bracket(df, new_i):
    return df.join(
        new_i, how='outer', on='i'
    ).select([
        pl.col(['country', 'year', 'reporting_level']).forward_fill(),
        pl.col('i').alias('bracket'),
        pl.col('headcount').fill_null(pl.lit(0))
    ])

def step5(res4):
    # 1. change CDF to shapes
    # 2. fix negative values
    # 3. fill more values until we have 500 groups
    res4_1 = res4.with_columns([
            pl.col('headcount').diff()
            .over(['country', 'year', 'reporting_level'])
        ]).drop_nulls().with_columns(pl.col('i') - 1)

    new_i = pl.Series(name='i', values=range(0, 500)).cast(pl.Int32).to_frame()

    return res4_1.groupby(
            ['country', 'year', 'reporting_level']
        ).apply(_group_fun).select([
            pl.col(['country', 'year', 'reporting_level', 'i']),
            pl.col('headcount_new').alias('headcount')
        ]).groupby(
            ['country', 'year', 'reporting_level']
        ).apply(lambda x: expand_bracket(x, new_i))


# %%
def func(x):
    """function to smooth a series"""
    run_smooth = smoothlib.run_smooth
    # run smoothing, based on standard deviation
    std = x.std()
    if std < 0.004:
        res = run_smooth(x, 40, 7)
        res = run_smooth(res, 30, 3)
        res = run_smooth(res, 20, 0)
        res = run_smooth(res, 20, 0)
        res = run_smooth(res, 10, 0)
        res = run_smooth(res, 10, 0)
    elif std <= 0.0045 and std > 0.004:
        res = run_smooth(x, 40, 5)
        res = run_smooth(res, 30, 2)
        res = run_smooth(res, 20, 0)
        res = run_smooth(res, 20, 0)
        res = run_smooth(res, 10, 0)
        res = run_smooth(res, 10, 0)
    elif std <= 0.0049 and std > 0.0045:
        res = run_smooth(x, 40, 3)
        res = run_smooth(res, 30, 2)
        res = run_smooth(res, 20, 0)
        res = run_smooth(res, 20, 0)
        res = run_smooth(res, 10, 0)
    elif std > 0.0049:
        res = run_smooth(x, 40, 2)
        res = run_smooth(res, 30, 1)
        res = run_smooth(res, 20, 0)
    # also, make sure it will sum up to 100%
    if res.min() < 0:
        res = res - res.min()
    res = res / res.sum()
    return pl.Series(res)
# %%
# df = _f(res5, country='IND', year=1999, reporting_level='national')
# # %%
# # df.select('headcount2').to_series().eval(func)
# df2 = df.with_columns(
#     pl.col('headcount').map(func)
# )
# # -> 360 ms
# # %% The time needed
# res5.select(['country', 'year', 'reporting_level']).unique().shape[0] * 400 / 1000 / 6 / 60
# # %%
# plt.plot(df.select('headcount'))
# plt.plot(df2.select('headcount'))
def group_func(df):
    return df.with_columns(
        pl.col('headcount').map(func)
    )
# %%
if __name__ == '__main__':
    res3 = pickle.load(open("./povcalnet_clean.pkl", 'rb'))
    res4 = step4(res3)
    res5 = step5(res4)

    with warnings.catch_warnings(record=False) as w:
        with get_context("spawn").Pool(8) as pool:
            todos = res5.partition_by(['country', 'year', 'reporting_level'])
            res6_lst = pool.map(group_func, todos)

    res6 = pl.concat(res6_lst)
    pickle.dump(res6, open('./povcalnet_smoothed.pkl', 'wb'))
    # TODO: add some checking, procedure images.
    plt.plot(_f(res4, country='BRA', year=2016, reporting_level='national').select('headcount').to_series().diff().drop_nulls(), alpha=.4)
    plt.plot(_f(res6, country='BRA', year=2016, reporting_level='national').select('headcount').to_series())
    plt.savefig("compare_smoothed.jpg")