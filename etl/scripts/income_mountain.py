"""create ddf files for income mountain

We will use the bridged version of income data to create it:

- The 105 bracket income mountain for global and country
- The 50 bracket income mountain for global, country and other regions
"""

# TODO: create a main function, make it more like a script

import os
import polars as pl
import pandas as pd


def _f(df, **kwargs):
    return df.filter(pl.all_horizontal([(pl.col(k) == v) for k, v in kwargs.items()]))


data = pl.read_parquet('./bridged_shapes.parquet')
# data

# data['bracket'].max()  # 997

# 1. resample the data and create shape in 105 brackets
# if we resample the data by adding each 10 brackets, we will end up with 100 brackets.
# we will use 105 brackets in total, to make sure all data
# including future updates can fit in

datalist = data.partition_by(['country', 'year'])


def resample_country(df, rmax=None, scale=10):  # will it work if scale = 20?
    country = df['country'][0]
    year = df['year'][0]
    df_ = df.with_columns(
        (pl.col('bracket') / scale).cast(pl.Int32).alias('bnew')
    ).group_by(['bnew'], maintain_order=True).agg(
        pl.col('population').sum()
    ).select(
        pl.col('bnew').alias('bracket'),
        pl.col('population')
    )
    if rmax is None:
        full_range = pl.Series('bracket', range(df_['bracket'].max()), dtype=pl.Int32).to_frame()
    else:
        full_range = pl.Series('bracket', range(rmax), dtype=pl.Int32).to_frame()

    return df_.join(
        full_range, on='bracket', how='outer'
    ).with_columns(
        pl.lit(country).alias('country'),
        pl.lit(year).alias('year'),
        pl.col('population').fill_null(0)
    ).select(['country', 'year', 'bracket', 'population'])


# use 105 brackets (0 - 104)
res = map(lambda x: resample_country(x, 105, 10), datalist)
res = pl.concat(res)

# res.select('bracket').max()  # 104
# res
# res.write_csv('other/country_shape_105.csv')

# see if we cut the shape at 50, what will it look like
# tot50plus = res.filter(
#     (pl.col('bracket') >= 50) & (pl.col('year') == 2020)
# ).select(
#     pl.col('population').sum()
# )

# tot = res.filter(
#     pl.col('year') == 2020
# ).select(
#     pl.col('population').sum()
# ).item()

# tot50plus  # 100 thousands
# tot50plus / tot * 100

# load povcalnet shape to compare
# data2 = pl.read_parquet('../build/population_500plus.parquet')


# create a function for creating shapes files in wide format
def join_str(d):
    return ','.join(map(str, d.values()))


# global shapes
res_gbl = res.group_by(['year', 'bracket'], maintain_order=True).agg(
    pl.col('population').sum()
)
res_gbl = res_gbl.select(
    pl.lit('world').alias('global'),
    pl.all()
)
res_gbl

os.makedirs('ddf/income_mountain', exist_ok=True)

# also create a global datapoint file with long format, for debugging.
res_gbl.select(
    pl.col('global'),
    pl.col('year').alias('time'),
    pl.col('bracket'),
    pl.col('population').alias('income_mountain_105bracket_shape_for_log')
).write_csv('ddf/income_mountain/ddf--datapoints--income_mountain_105brackets--by--global--time.csv')

# plot 2022
# res_gbl_22 = res_gbl.filter(
#     (pl.col('year') == 2022) & (pl.col('bracket') > 40)
# ).sort('bracket')
# plt.plot(res_gbl_22['bracket'], res_gbl_22['population'])
# plt.show()

res_pivot_gbl = res_gbl.pivot(values='population', index=['global', 'year'], columns='bracket', aggregate_function=None)
res_pivot_gbl = res_pivot_gbl.fill_null(0)
res_pivot_gbl
out_gbl = res_pivot_gbl.select(
    pl.col('global'),
    pl.col('year').alias('time'),
    pl.struct(pl.col(map(str, (range(0, 105))))).map_elements(join_str).alias('income_mountain_105bracket_shape_for_log')
)
out_gbl

out_gbl.write_csv('ddf/income_mountain/ddf--datapoints--income_mountain_105bracket_shape_for_log--by--global--time.csv')

res_pivot = res.pivot(values='population', index=['country', 'year'], columns='bracket', aggregate_function=None)
res_pivot = res_pivot.fill_null(0)
out = res_pivot.select(
    pl.col('country'),
    pl.col('year').alias('time'),
    pl.struct(pl.col(map(str, (range(0, 105))))).map_elements(join_str).alias('income_mountain_105bracket_shape_for_log')
)
out

out.write_csv('ddf/income_mountain/ddf--datapoints--income_mountain_105bracket_shape_for_log--by--country--time.csv')

# so we will drop those bracket > 49 to create a version of world shape in 50 groups.
res50 = res.filter(
    pl.col('bracket') < 50
)
# res50

res50_pivot = res50.pivot(values='population', index=['country', 'year'], columns='bracket', aggregate_function=None)
res50_pivot = res50_pivot.fill_null(0)
out50 = res50_pivot.select(
    pl.col('country'),
    pl.col('year').alias('time'),
    pl.struct(pl.col(map(str, (range(0, 50))))).map_elements(join_str).alias('income_mountain_50bracket_shape_for_log')
)
# out50
out50.write_csv('ddf/income_mountain/ddf--datapoints--income_mountain_50bracket_shape_for_log--by--country--time.csv')

# 2. for 50 bracket data, we also need other regions
# below are taken from an older scripts where we still use pandas.
# TODO: maybe update following to use polars.
max_heights = dict()
max_heights['country'] = res50.group_by('country').agg(pl.col('population').max())

# max_heights


def concat_values(ser):
    res = ser.astype(int).unstack().apply(lambda r: r.astype(str).str.cat(sep=','), axis=1)
    res.name = 'income_mountain_50bracket_shape_for_log'
    return res


# income groups
wb_groups = pd.read_csv('../build/source/gapminder/wb_income_groups.csv')

wb = wb_groups[['geo', 'time', "4 income groups (WB)"]].copy()

wb.columns = ['geo', 'time', 'level']

on_income = pd.read_csv('../build/source/gapminder/on_income_groups.csv')
on_income = on_income[['geo', 'time', 'Income levels (GM)']].dropna()
on_income.columns = ['geo', 'time', 'level']

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

on = on_income.copy()
on['level'] = on['level'].map(lambda x: on_map[x])
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

df = res50.to_pandas()
df.columns = ['geo', 'time', 'income_bracket_50', 'population']

income_groups = income_groups.sort_index()
income_3groups = income_3groups.sort_index()

_d = income_groups['level'].to_dict()
_lv = income_groups['level']

ig = [_d[(row['geo'], row['time'])] for _, row in df.iterrows()]

df['income_groups'] = ig

res_ig = df.groupby(by=['income_groups', 'time', 'income_bracket_50'])['population'].sum()

res_ig
# res_ig.loc['high_income', 2021].plot()
# plt.show()

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

# others
countries = pd.read_csv('../build/source/datasets/ddf--open_numbers/ddf--entities--geo--country.csv')

# global
df['global'] = 'world'
res_glob = df.groupby(by=['global', 'time', 'income_bracket_50'])['population'].sum()
max_heights['global'] = res_glob.groupby('global').max()
res_glob = concat_values(res_glob)
res_glob.to_csv('ddf/income_mountain/ddf--datapoints--income_mountain_50bracket_shape_for_log--by--global--time.csv')
# res_glob.loc['world', '2022'].plot()
# plt.show()

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
max_heights['country'] = max_heights['country'].to_pandas().set_index('country')['population']

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

print("Done!")
