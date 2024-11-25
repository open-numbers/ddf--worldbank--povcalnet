# encoding: utf-8

"""Convert income mountain into normalized shapes

by moving mountains so that the group 0 will contains the mean income
"""

# %%
import os
import sys

import numpy as np
import polars as pl

import bracketlib

import matplotlib.pyplot as plt
import seaborn as sns
# %%
# settings for display images
sns.set_context('notebook')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (7, 4)
plt.rcParams['figure.dpi'] = 144
# %%
income_file = 'source/gapminder/mean_income.csv'
income = pl.read_csv(income_file).select(
    pl.col('geo').alias('country'),
    pl.col('time').cast(pl.Int32).alias('year'),
    pl.col('Average daily income per capita').alias('income')
)


# %%
def _f(df, **kwargs):
    return df.filter(pl.all_horizontal([(pl.col(k) == v) for k, v in kwargs.items()]))


def get_income(c, y):
    inc = _f(income, country=c, year=y)
    if inc.is_empty():
        return None
    return inc.select('income').item()


def move_mean(df: pl.DataFrame):
    c = df.select('country')[0].item()
    y = df.select('year')[0].item()
    inc = get_income(c, y)
    if inc:
        # bracket_from_income(8192, 0.04) -> should be 500
        brack = bracketlib.bracket_from_income(inc, bracket_step=0.04)
        return df.with_columns([
            pl.col('bracket') - brack
        ])
    else:
        print(c, y)
        return df.select([
            pl.exclude('headcount')
        ]).with_columns(
            pl.lit(np.nan).alias('headcount')
        )


def step8(res7):
    print("below are country/year which existed in povcalnet but not gapminder income data:")
    # move the shapes to make the group with mean income to be group 0
    return res7.group_by('country', 'year', 'reporting_level').map_groups(move_mean)


# %%
if __name__ == '__main__':
    res7 = pl.read_parquet('./povcalnet_smoothed.parquet')
    # FIXME: make sure res7's year column is int
    res7 = res7.with_columns(
        pl.col('year').cast(pl.Int32)
    )
    res8 = step8(res7)

    res8.write_parquet('mean_central_shapes.parquet')

    df = _f(res8, country='syr', year=2022, reporting_level='n')
    print('example shape:')
    print(df)
    plt.plot(df.select('bracket'), df.select('headcount'))
    plt.savefig("mean_central_shape.jpg")
