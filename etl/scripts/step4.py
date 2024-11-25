# encoding: utf-8

"""estimate all income mountains outside the povcalnet data range

we calculate following shapes

- average shape of neighbour shapes
- nearest povcalnet shape

and we use a weighted sum of above 2 shapes to calculate the estimated shape
of given country/year.
"""

# %%
import os
import sys

import numpy as np
import polars as pl
import pandas as pd
import json
from multiprocessing import get_context
from functools import partial

import bracketlib
import constants
import step3

import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# %%
# settings for display images
sns.set_context('notebook')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (7, 4)
plt.rcParams['figure.dpi'] = 144

# source files needed:
gini_file = 'source/gapminder/gini.csv'
neighbours_file = 'neighbours_list.json'


def _f(df, **kwargs):
    return df.filter(pl.all_horizontal([(pl.col(k) == v) for k, v in kwargs.items()]))


def select_shape(df):
    for level in 'naur':
        res = df.filter(
            pl.col('reporting_level') == level
        ).select(
            pl.exclude('reporting_level')
        )
        if not res.is_empty():
            return res


def normalize_shape(shape):
    return shape.with_columns(
        pl.col('headcount') / pl.col('headcount').sum()
    )


def get_average_shape(known_shapes, neighbours_list):
    other = pl.DataFrame(
        [{'country': x[0], 'year': x[1]} for x in neighbours_list],
        schema={'country': pl.Utf8, 'year': pl.Int32}
    )
    res = known_shapes.join(other, on=['country', 'year'], how='inner')
    res = res.group_by('bracket').agg(
        pl.col('headcount').sum() / 50
    ).sort('bracket')
    return normalize_shape(res)


def get_nearest_known_shape(country, year, known_shapes):
    country_df = known_shapes.filter(pl.col('country') == country)
    return country_df.with_columns(
        (pl.col('year') - year).abs().alias('year_diff')
    ).filter(
        pl.col('year_diff') == pl.col('year_diff').min()
    ).select(pl.col(['bracket', 'headcount']))


def shape_to_mountain_robin(shape, income):
    bracket = bracketlib.bracket_from_income(income, 0.04, integer=False)
    return shape.with_columns(
        pl.col('bracket') + bracket
    )


def get_income_gini(country, year, income_gini):
    i = _f(income_gini, country=country, year=year).select('income').item()
    g = _f(income_gini, country=country, year=year).select('gini').item()
    return i, g


def get_estimated_shape(country, year, income, known_shapes, neighbours_list):
    wpov, was = constants.all_weights[year]
    nearest_shape = get_nearest_known_shape(country, year, known_shapes)
    if nearest_shape.is_empty():
        return None

    if wpov == 1:
        mixed_shape = nearest_shape.clone()
    elif wpov == 0:
        mixed_shape = get_average_shape(known_shapes, neighbours_list)
    else:
        average_shape = get_average_shape(known_shapes, neighbours_list)
        mixed_shape = average_shape.join(
            nearest_shape, on=['bracket'], how='full', suffix='_povcalnet', coalesce=True
        ).select(
            pl.col('bracket'),
            (pl.col('headcount').fill_null(0) * was + pl.col('headcount_povcalnet').fill_null(0) * wpov).alias('headcount')
        )
    # to_check = mixed_shape.filter(pl.col('headcount').is_null())
    # if not to_check.is_empty():
    #     print(to_check)
    #     print(f"wpov: {wpov}, was: {was}")
    #     raise ValueError(f"should not have duplicated bracket: {country}, {year}")
    return normalize_shape(mixed_shape.sort('bracket'))


def process_step8(i,
                  unknown_countries,
                  known_country_year,
                  all_neighbours_json,
                  income_gini,
                  known_shapes):
    country, year = i
    neis = all_neighbours_json[country][str(year)]['neighbours']
    neis = [[x[0], int(x[1])] for x in neis]

    if country in unknown_countries:
        # only use Average Shapes
        income, _ = get_income_gini(country, year, income_gini)
        average_shape = get_average_shape(known_shapes, neis)
        res = shape_to_mountain_robin(average_shape, income)
        return res.with_columns(
            pl.lit(country).alias('country'),
            pl.lit(year).alias('year')
        ).select(
            ['country', 'year', 'bracket', 'headcount']
        ).sort('bracket').filter(pl.col('headcount') > 0)
    if [country, year] not in known_country_year:
        # we have povcalnet shapes for this country, but not this year.
        income, _ = get_income_gini(country, year, income_gini)
        res_shape = get_estimated_shape(country, year, income, known_shapes, neis)
        res = shape_to_mountain_robin(res_shape, income)
        return res.with_columns(
            pl.lit(country).alias('country'),
            pl.lit(year).alias('year')
        ).select(
            ['country', 'year', 'bracket', 'headcount']
        ).sort('bracket').filter(pl.col('headcount') > 0)
    # otherwise, we have povcalnet shapes so this is not needed
    return None


if __name__ == '__main__':
    income = step3.income
    gini = pl.read_csv(gini_file).select(
        pl.col('geo').alias('country'),
        pl.col('time').cast(pl.Int32).alias('year'),
        pl.col('gini_2100').alias('gini')
    )
    income_gini = gini.join(income,
                            on=['country', 'year'],
                            how='full',
                            coalesce=True).drop_nulls()

    # load neighbours
    fp = open(neighbours_file, 'r')
    jsonstring = fp.read()
    all_neighbours_json = json.loads(jsonstring)

    # load shapes
    res8 = pl.read_parquet('./mean_central_shapes.parquet')

    # 1. only keep one reporting level
    known_shapes = list()
    for x in res8.partition_by(['country', 'year']):
        known_shapes.append(select_shape(x))
    known_shapes = pl.concat(known_shapes)
    known_shapes_lazy = known_shapes.lazy()

    # 2. there are 3 cases
    # 2.1 povcalnet shapes: just load from known shapes
    # 2.2 the country is in povcalnet: was * average shapes + wpov * povcalnet shapes
    # 2.3 the country is not in povcalnet: use average shapes only

    # build all average_shapes
    # average_shapes = all_neighbours_json.copy()

    # for c, ys in all_neighbours_json.items():
    #     # average_shapes[c] = list()
    #     for y, neis in ys.items():
    #         neis_ = [[x[0], int(x[1])] for x in neis['neighbours']]
    #         avg_shape = get_average_shape(known_shapes_lazy, neis_).with_columns(
    #             pl.lit(int(y)).alias('year')
    #         )
    #         average_shapes[int(y)] = avg_shape.collect()

    # build lists about existing ones and missing ones
    unknown_list = list()
    all_countries = income_gini.select(['country']).unique()
    known_countries = known_shapes.select(['country']).unique()
    known_list = known_shapes.select(['country', 'year']).unique().to_numpy().tolist()
    for c in all_countries.join(known_countries, on=['country'], how='anti').to_series():
        unknown_list.append(c)
        print(c, "not in povcalnet")

    run = partial(process_step8,
                  unknown_countries=unknown_list,
                  known_country_year=known_list,
                  all_neighbours_json=all_neighbours_json,
                  income_gini=income_gini,
                  known_shapes=known_shapes)

    todos = income_gini.select(['country', 'year']).unique().to_numpy().tolist()

    # test if the constants are collectly defined
    for country, year in todos:
        wpov, was = constants.all_weights[year]

    # # with get_context("spawn").Pool(2) as pool:
    # #     estimated = pool.map(run, todos[:8])
    # print(len(todos))
    poolsize = 2

    with warnings.catch_warnings(record=False) as w:
        with get_context("spawn").Pool(poolsize) as pool:
            estimated = pool.map(run, todos)

    estimated_df = pl.concat([x for x in estimated if x is not None])

    estimated_df.write_parquet('./estimated_mountains.parquet')
