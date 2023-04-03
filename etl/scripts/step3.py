# encoding: utf-8

"""Convert income mountain into normalized shapes

by moving mountains so that the group 0 will contains the mean income
"""

# %%
import os
import sys

import numpy as np
import polars as pl
import pickle

import etllib

import matplotlib.pyplot as plt
import seaborn as sns
# %%
# settings for display images
sns.set_context('notebook')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (7, 4)
plt.rcParams['figure.dpi'] = 144
# %%
income_file = '../notebooks/temporary mean income 20230214 - Sheet1.csv'
# %%
def _f(df, **kwargs):
    return df.filter(pl.all([(pl.col(k) == v) for k, v in kwargs.items()]))
# %%
# df = _f(res6, country='USA', year=2012, reporting_level='national')
# print(df.tail())
# plt.plot(df.select('headcount'))
# %%
def step7(res6: pl.DataFrame):
    # rename things
    mapping = {'national': 'n', 'rural': 'r', 'urban': 'u'}

    # TODO: change headcount -> population_percentage?
    return res6.with_columns(
        pl.col('country').str.to_lowercase().str.replace("xkx", "kos"),
        pl.col('reporting_level').map_dict(mapping)
    )

#
def tmp_preprocess_income_file(income_file):
    # this is a temporary function to convert the income file
    # to a tidy format
    income = pl.read_csv(income_file)
    return income.select(
        pl.exclude('name')
    ).melt(
        id_vars='geo', value_vars=income.columns[2:], variable_name='year', value_name='income'
    ).select([
        pl.col('geo').alias('country'),
        pl.col('year').cast(pl.Int32),
        pl.col('income')
    ])

income = tmp_preprocess_income_file(income_file)

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
        # etllib.bracket_number_from_income_robin(8192) -> should be 500
        brack = etllib.bracket_number_from_income_robin(inc)
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

def step8(step7):
    # move the shapes to make the group with mean income to be group 0
    return res7.groupby(['country', 'year', 'reporting_level']).apply(move_mean)

# %%
if __name__ == '__main__':
    res6 = pickle.load(open("./povcalnet_smoothed.pkl", 'rb'))
    res7 = step7(res6)
    # FIXME: move step 7 to step2
    pickle.dump(res7, open('povcalnet_smoothed_2.pkl', 'wb'))

    # save all available povcalnet years
    povcal_country_year = res7.select(['country', 'year']).unique()
    povcal_country_year.write_csv('./povcal_country_year.csv')
    res8 = step8(res7)

    pickle.dump(res8, open('mean_central_shapes.pkl', 'wb'))

    df = _f(res8, country='ago', year=1981, reporting_level='n')
    plt.plot(df.select('bracket'), df.select('headcount'))
    plt.savefig("mean_central_shape.jpg")
