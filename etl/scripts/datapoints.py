# -*- coding: utf-8 -*-

"""create income mountain datapoints for countries/regions.
"""

import os

import numpy as np
import polars as pl
import pandas as pd
import pickle

import matplotlib.pyplot as plt


data = pickle.load(open('../build/population_500plus.pkl', 'rb'))


bracket_range = pl.Series('bracket', range(500), dtype=pl.Int32).to_frame()

bracket_range = bracket_range.to_pandas()
bracket_range


data500 = data.to_pandas()

data500


data500 = data500.set_index(['country', 'year', 'bracket'])

# data500.set_index('bracket').join(bracket_range.set_index('bracket'), on='bracket', how='inner')


def resample_to_50(ser):
    res = pd.Series([x.sum() for x in np.split(ser.values, 50)])
    res.index.name = 'bracket'
    return res


def run_resample_to_int(x):
    ser_copy = x.droplevel([0, 1]).loc[:499]
    ser_copy = ser_copy.reindex(range(500)).fillna(0)
    return resample_to_50(ser_copy)


def row_to_str(row):
    row_str = ','.join(row['population'].astype(int).astype(str))
    return row_str


res = data500.groupby(['country', 'year'])['population'].apply(run_resample_to_int)

res.loc['zmb', 1964]


os.makedirs('ddf', exist_ok=True)


# reshape
# res2 = res.to_frame().unstack()

res2 = res.to_frame().groupby(['country', 'year'], group_keys=True).apply(row_to_str)

res2.index.names = ['country', 'time']
res2.name = 'income_mountain_50bracket_shape_for_log'

os.makedirs('ddf/income_mountain', exist_ok=True)

res2.to_csv('ddf/income_mountain/ddf--datapoints--income_mountain_50bracket_shape_for_log--by--country--time.csv')


def concat_values(ser):
    res = ser.astype(int).unstack().apply(lambda r: r.astype(str).str.cat(sep=','), axis=1)
    res.name = 'income_mountain_50bracket_shape_for_log'
    return res


# below: create groups data

max_heights = dict()

max_heights['country'] = res.groupby('country').max()



wb_groups = pd.read_csv('../build/source/gapminder/wb_income_groups.csv')

wb = wb_groups[['geo', 'time', "4 income groups (WB)"]].copy()

wb.columns = ['geo', 'time', 'level']

on_income = pd.read_csv('../build/source/gapminder/on_income_groups.csv')
on_income = on_income[['geo', 'time', 'Income levels (GM)']].dropna()
on_income.columns = ['geo', 'time', 'level']

on = on_income.copy()
on['level'] = on['level'].map(lambda x: on_map[x])


on_map = {'Level 1': 'l1',
          'Level 2': 'l2',
          'Level 3': 'l3',
          'Level 4': 'l4',
         }

wb_map = {'Low income': 'l1',
          'Lower middle income': 'l2',
          'Upper middle income': 'l3',
          'High income': 'l4',
         }


wb['level'] = wb['level'].map(lambda x: wb_map[x])


wb = wb.set_index(['geo', 'time'])
on = on.set_index(['geo', 'time'])


on.update(wb)

income_groups = on.copy()
income_3groups = income_groups.copy()

concept_id_map = {'l1': 'low_income',
                  'l2': 'lower_middle_income',
                  'l3': 'upper_middle_income',
                  'l4': 'high_income'
                 }

concept_id_map3 = {'l1': 'low_income',
                   'l2': 'middle_income',
                   'l3': 'middle_income',
                   'l4': 'high_income'
                 }


income_groups['level'] = income_groups['level'].map(lambda x: concept_id_map[x])
income_3groups['level'] = income_3groups['level'].map(lambda x: concept_id_map3[x])


df = res.copy()

df = df.reset_index()

df.columns = ['geo', 'time', 'income_bracket_50', 'population']

income_groups = income_groups.sort_index()
income_3groups = income_3groups.sort_index()

_d = income_groups['level'].to_dict()
_lv = income_groups['level']

ig = [_d[(row['geo'], row['time'])] for _, row in df.iterrows()]

df['income_groups'] = ig

res_ig = df.groupby(by=['income_groups', 'time', 'income_bracket_50'])['population'].sum()

res_ig
res_ig.loc['high_income', 2021].plot()
plt.show()


max_heights['income_groups'] = res_ig.groupby('income_groups').max()

res_ig = concat_values(res_ig)

res_ig.to_csv('ddf/income_mountain/ddf--datapoints--income_mountain_50bracket_shape_for_log--by--income_groups--time.csv')


# income 3 groups

_d = income_3groups['level'].to_dict()
ig3 = [_d[(row['geo'], row['time'])] for _, row in df.iterrows()]

df['income_3groups'] = ig3


res_ig3 = df.groupby(by=['income_3groups', 'time', 'income_bracket_50'])['population'].sum()
max_heights['income_3groups'] = res_ig3.groupby('income_3groups').max()

res_ig3 = concat_values(res_ig3)

res_ig3.to_csv('ddf/income_mountain/ddf--datapoints--income_mountain_50bracket_shape_for_log--by--income_3groups--time.csv')


res_ig3.at[('middle_income', 2050)]


# others

countries = pd.read_csv('../source/fixtures/ddf--open_numbers/ddf--entities--geo--country.csv')


# global
df['global'] = 'world'
res_glob = df.groupby(by=['global', 'time', 'income_bracket_50'])['population'].sum()
max_heights['global'] = res_glob.groupby('global').max()
res_glob = concat_values(res_glob)
res_glob.to_csv('ddf/income_mountain/ddf--datapoints--income_mountain_50bracket_shape_for_log--by--global--time.csv')


# let's use a loop!

for k in [
        'g77_and_oecd_countries',
        'landlocked',
        'main_religion_2008',
        'unhcr_region',
        'unicef_region',
        'un_sdg_ldc',
        'un_sdg_region',
        'west_and_rest',
        'world_4region',
        'world_6region']:
    _d = countries.set_index('country')[k].dropna().to_dict()
    if k == 'g77_and_oecd_countries':
        # this one missing in open_numbers ontology
        _d['ssd'] = 'g77'
    missing = set()

    def _f(x):
        try:
            return _d[x]
        except KeyError:
            missing.add(x)
            return None

    df[k] = df['geo'].map(_f)
    print(k, ":", missing)

    res_grp = df.dropna(subset=[k]).groupby(by=[k, 'time', 'income_bracket_50'])['population'].sum()
    max_heights[k] = res_grp.groupby(k).max()

    res_grp = concat_values(res_grp)
    res_grp.to_csv(f'ddf/income_mountain/ddf--datapoints--income_mountain_50bracket_shape_for_log--by--{k}--time.csv')


# now also update the max heights for all groups!
for k in ['country', 'income_groups',
          'income_3groups', 'g77_and_oecd_countries', 'global', 'landlocked', 'main_religion_2008',
          'unhcr_region', 'unicef_region', 'un_sdg_ldc', 'un_sdg_region',
          'west_and_rest', 'world_4region', 'world_6region']:
    fn = f'ddf/ddf--entities--geo--{k}.csv'
    ent = pd.read_csv(fn, dtype=str).set_index(k)
    try:
        mh = max_heights[k]
        ent.loc[mh.index, 'income_mountain_50bracket_max_height_for_log'] = mh.astype(str)
        ent.to_csv(fn)
    except KeyError:
        print(k, "has error")
        raise
