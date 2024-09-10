# -*- coding: utf-8 -*-

"""Load and clean up the povcalnet shapes

1. load all source files
2. merge all files into one dataframe and add an `i` column
3. check and fill null values
"""


# %%
import os
import sys
import pickle

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
# settings for files
source_dir = 'source/povcalnet'
usecols = [
    'country_code', 'country_name', 'reporting_level', 'reporting_year',
    'headcount', 'reporting_pop', 'mean'
]
MAX_BRACKET = 460
# %%
# I have run the following to determine the data types
# df = pl.read_csv(os.path.join(source_dir, "0000.csv"),
#                  infer_schema_length=None,
#                  columns=usecols)
# dict(zip(df.columns, df.dtypes))

dtypes = {'region_name': pl.Utf8,
          'region_code': pl.Utf8,
          'country_name': pl.Utf8,
          'country_code': pl.Utf8,
          'reporting_year': pl.Int64,
          'reporting_level': pl.Utf8,
          'survey_acronym': pl.Utf8,
          'survey_coverage': pl.Utf8,
          'survey_year': pl.Utf8,
          'welfare_type': pl.Utf8,
          'survey_comparability': pl.Utf8,
          'comparable_spell': pl.Utf8,
          'poverty_line': pl.Float64,
          'headcount': pl.Float64,
          'poverty_gap': pl.Float64,
          'poverty_severity': pl.Float64,
          'watts': pl.Float64,
          'mean': pl.Float64,
          'median': pl.Float64,
          'mld': pl.Float64,
          'gini': pl.Float64,
          'polarization': pl.Float64,
          'decile1': pl.Float64,
          'decile2': pl.Float64,
          'decile3': pl.Float64,
          'decile4': pl.Float64,
          'decile5': pl.Float64,
          'decile6': pl.Float64,
          'decile7': pl.Float64,
          'decile8': pl.Float64,
          'decile9': pl.Float64,
          'decile10': pl.Float64,
          'cpi': pl.Utf8,
          'ppp': pl.Float64,
          'reporting_pop': pl.Float64,
          'reporting_gdp': pl.Float64,
          'reporting_pce': pl.Float64,
          'is_interpolated': pl.Boolean,
          'distribution_type': pl.Utf8,
          'estimation_type': pl.Utf8}


# %%
def load_file_preprocess(filename):
    df = pl.read_csv(filename,
                     columns=usecols,
                     dtypes=dtypes,
                     null_values=['NA'],
                     )
    df = df.rename(
        {'country_code': 'country', 'reporting_year': 'year'}
    )
    # check if there are duplicates
    dups = df.filter(
        df.select(['country', 'reporting_level', 'year']).is_duplicated()
    )
    if not dups.is_empty():
        print(f"{filename} has duplicated entries")
        print(dups)
    return df


# %%
def step1():
    # read all source file into a dictionary
    res = dict()
    for f in os.listdir(source_dir):
        if f.endswith('.csv'):
            fn = f.split('.')[0]
            bracket = fn.lstrip('0')
            if bracket == '':
                bracket = 0
            else:
                bracket = int(bracket)
            if bracket <= MAX_BRACKET:
                res[bracket] = load_file_preprocess(os.path.join(source_dir, f))
    return res


# %%
def step2(res1):
    # insert a new index level, merge all dataframes
    res = list()
    for i, df in res1.items():
        df_new = df.with_columns([pl.lit(i).alias("i")])
        df_new = df_new.select([
            pl.col(['country', 'year', 'reporting_level', 'i', 'headcount'])
        ])
        res.append(df_new)
    return pl.concat(res)


# %%
def run_fill(df: pl.Series, fill_until: int):
    ser = df.to_pandas()
    ser.iloc[fill_until:] = 1
    ser = ser.interpolate(method='cubic', order=2)
    ser[ser>1] = 1
    return pl.from_pandas(ser)


# %%
def run_fill_df(df_input: pl.DataFrame, fillna_til):
    df = df_input.clone()
    if df.select('headcount').null_count().item() > 0:
        country = df.select('country').unique().item()
        level = df.select('reporting_level').unique().item()
        til = fillna_til[(country, level)]
        return df.with_columns([
            pl.col('headcount').map_batches(lambda x: run_fill(x, til))
        ])
    else:
        return df


def step3(res2):
    print("fixing missing values...")
    # We assumed that all missing values are in the tail,
    # so let's add a checking for that
    _missing = res2.filter(
        pl.col('headcount').is_null()
    ).select(pl.col(['country', 'year', 'reporting_level'])).unique()
    # FIXME: check if there are indeed some missing values.
    for row in _missing.to_dicts():
        lastcount = res2.filter(
            (pl.col('country') == row['country']) &
            (pl.col('year') == row['year']) &
            (pl.col('reporting_level') == row['reporting_level']) &
            (pl.col('headcount').is_not_null())
        ).select(
            pl.col("i").last()
        ).item()
        if lastcount < 0.8:
            print("[WARN] please double check: ", row)
    else:
        print("all missing values are in the tail")

    # Now fill in the nulls
    # steps:
    # 1. find out which years are missing
    # 2. find the bracket where we hit the maxinum headcount
    # 3. calculate the hit point we should use for missing data
    _missing_years = _missing.group_by("country", "reporting_level").agg([
        pl.col('year').min().alias('min_year'),
        pl.col('year').max().alias('max_year')
    ])
    # fillna_til: the hit point
    fillna_til = dict()
    # find the hit point for each missing country
    for row in _missing_years.to_dicts():
        if row['min_year'] == 1981:
            # we have check that all missing years starts from 1981, which
            # is the beginning of povcalnet data. Thus we don't have data for 1980.
            # and only use the max_year for calculation
            fst_avail_year = row['max_year'] + 1
            fst_max_bracket = res2.filter(
                (pl.col('country') == row['country']) &
                (pl.col('year') == fst_avail_year) &
                (pl.col('reporting_level') == row['reporting_level'])
            ).unique(subset=['headcount']).select(pl.col('i').last()).item()
            fillna_til[(row['country'], row['reporting_level'])] = fst_max_bracket
        else:
            # if min_year > 1981, we can find the hit point
            # using both min year and max year.
            raise NotImplementedError("min year > 1981")

    res3 = (
        res2.group_by('country', 'year', 'reporting_level')
            .map_groups(lambda x: run_fill_df(x, fillna_til))
            .sort(['country', 'year', 'reporting_level', 'i']))
    return res3


# %%
if __name__ == "__main__":
    # 1. load all source files
    res1 = step1()
    # 2. merge all files into one dataframe and add an `i` column
    res2 = step2(res1)
    # 3. filling null values
    res3 = step3(res2)
    # some checking
    assert res3.shape == res2.shape
    assert res3.filter(pl.col('headcount') > 1).is_empty()
    assert res3.filter(pl.col('headcount') < 0).is_empty()
    # save result to parquet
    res3.write_parquet('./povcalnet_clean.parquet')
    print("please remember to update povcalnet_clean.parquet under etl/source/")
