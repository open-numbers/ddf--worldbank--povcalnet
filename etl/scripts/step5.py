# -*- coding: utf-8 -*-

"""finalize the result

calculate population_percentage and population by country/year/income_group
"""

# %%
import os
import sys

import numpy as np
import polars as pl
import pandas as pd
import pickle
import json
from multiprocessing import get_context
from functools import partial

import etllib
import constants
import step3

import matplotlib.pyplot as plt
import seaborn as sns

# %%
# settings for display images
sns.set_context('notebook')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (7, 4)
plt.rcParams['figure.dpi'] = 144


def _f(df, **kwargs):
    return df.filter(pl.all([(pl.col(k) == v) for k, v in kwargs.items()]))


def select_shape(df):
    for level in 'naur':  # n: national, a: aggregate, u: urban, r: rural
        res = df.filter(
            pl.col('reporting_level') == level
        ).select(
            pl.exclude('reporting_level')
        )
        if not res.is_empty():
            return res


# reindex estimated mountains' bracket to int
def resample_to_int(df, cut=True):
    if cut:
        newIndex = pl.DataFrame({'bracket': range(500)})
    else:
        newIndex = pl.DataFrame({
            'bracket': range(
                int(df['bracket'].max())
            )})

    newIndex = newIndex.with_columns(pl.col('bracket').cast(pl.Float64))
    country = df['country'].unique()[0]
    year = df['year'].unique()[0]

    res = df.join(newIndex, on='bracket', how='outer').sort('bracket')
    res = res.with_columns(
        pl.col('country').fill_null(country),
        pl.col('year').fill_null(year),
        pl.col('headcount').interpolate().fill_null(0)
    )
    return newIndex.join(res, on='bracket', how='inner').select(
        pl.col(['country', 'year']),
        pl.col('bracket').cast(pl.Int64),
        pl.col('headcount')
    ).sort(['bracket'])


if __name__ == '__main__':
    # load data
    povcalnet = pickle.load(open('./povcalnet_smoothed_2.pkl', 'rb'))
    estimated = pickle.load(open('./estimated_mountains.pkl', 'rb'))

    # only keep one reporting level. They are mostly `national` but there are
    # some countries we will use urban.
    povcalnet = povcalnet.groupby(['country', 'year']).apply(select_shape)
    # resample estimated mountains to use integer brackets
    est = estimated.groupby(['country', 'year']).apply(lambda x: resample_to_int(x, cut=False))
    # check if something abnormal
    assert est.filter(pl.col('headcount') < 0).is_empty()

    # use same data types
    povcalnet = povcalnet.with_columns(
        pl.col('year').cast(pl.Int32),
        pl.col('bracket').cast(pl.Int32),
    )
    est = est.with_columns(
        pl.col('year').cast(pl.Int32),
        pl.col('bracket').cast(pl.Int32),
    )

    povcal_and_est = pl.concat([est, povcalnet]).sort(['country', 'year', 'bracket'])

    # product 1: population percentage
    pickle.dump(povcal_and_est, open('./population_percentage_500plus.pkl', 'wb'))

    # then load population data and create population numbers datapoint
    # FIXME: keep one copy in source folder, or download from url?
    pop_file = '../../../ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--population_total--by--geo--time.csv'
    pop = pl.read_csv(pop_file)

    pop.columns = ['country', 'year', 'population_total']
    pop = pop.with_columns(pl.col('year').cast(pl.Int32))

    res = povcal_and_est.join(pop, on=['country', 'year'], how='inner').with_columns(
        (pl.col('headcount') * pl.col('population_total')).alias('population')
    ).select(
        pl.col(['country', 'year', 'bracket']),
        pl.col('population').round(0).cast(pl.Int64)
    )
    # check missing ones
    # KOS is expected to be missing
    # Out[34]:
    # shape: (20, 2)
    # ┌─────────┬──────┐
    # │ country ┆ year │
    # │ ---     ┆ ---  │
    # │ str     ┆ i32  │
    # ╞═════════╪══════╡
    # │ kos     ┆ 2000 │
    # │ kos     ┆ 2001 │
    # │ kos     ┆ 2002 │
    # │ kos     ┆ 2003 │
    # │ ...     ┆ ...  │
    # │ kos     ┆ 2016 │
    # │ kos     ┆ 2017 │
    # │ kos     ┆ 2018 │
    # │ kos     ┆ 2019 │
    # └─────────┴──────┘
    missing = povcal_and_est.select(['country', 'year']).unique().join(
        pop.select(['country', 'year']), on=['country', 'year'], how='anti')
    if not missing.is_empty():
        print('some countries do not have total population data:')
        for rec in missing.to_dicts():
            print(rec['country'], rec['year'])

    # assert no null in data
    assert res.filter(pl.col('population').is_null()).is_empty()

    # product 2:
    pickle.dump(res, open('./population_500plus.pkl', 'wb'))


# # check global shapes
# df = _f(res, year=2022)
# df
# df2 = df.groupby(['bracket']).agg(
#     pl.col('population').sum()
# ).sort('bracket')

# _, ax = plt.subplots(1, 1)
# plt.plot(df2['bracket'], df2['population'])
# ax.set_yscale('log')
# plt.show()

# TODO: missing steps
# 1. create the flattened datapoint
# 2. create datapoint by regions
