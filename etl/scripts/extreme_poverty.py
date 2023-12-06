"""calculate the extreme poverty rate
"""
import os
import numpy as np
import polars as pl
import pandas as pd
import bracketlib

data = pl.read_parquet('./bridged_shapes.parquet')

data


# let's try global first
data_gbl = data.group_by(['year', 'bracket']).agg(pl.col('population').sum())
data_gbl

data_gbl

step = bracketlib.get_bracket_step(500)

# where is 2.15/day

xloc = np.log10(2.15)
xdf = pl.DataFrame({'bracket': [xloc]})

# 2020
data_gbl.filter(
    pl.col('year') == 1981
).sort('bracket').with_columns(
    pl.col('population').cumsum() / pl.col('population').sum()
).filter(
    pl.col('bracket').is_in([201, 202])
).with_columns(
    pl.col('bracket').map_elements(lambda x: np.log10(bracketlib.income_from_bracket(x, step, integer=False)))
)


# get all global items
def get_epov_rates(df):
    df_ = df.sort('bracket').with_columns(
        pl.col('population').cumsum() / pl.col('population').sum()
    ).filter(
        pl.col('bracket').is_in([201, 202])
    ).with_columns(
        pl.col('bracket').map_elements(lambda x: np.log10(bracketlib.income_from_bracket(x, step, integer=False)))
    )
    x = df_['bracket'].to_numpy()
    y = df_['population'].to_numpy()

    xnew = np.array([xloc])
    ynew = np.interp(xnew, x, y)
    return ynew[0] * 100


get_epov_rates(data_gbl.filter(
    pl.col('year') == 1981
))

global_epov = list()
for y in data_gbl['year'].unique():
    _df = data_gbl.filter(
        pl.col('year') == y
    )
    global_epov.append({'year': y, 'epov': get_epov_rates(_df)})


gbl_epov_rates = pl.from_records(global_epov)

os.makedirs('./ddf/poverty_rates/', exist_ok=True)

gbl_epov_rates = gbl_epov_rates.select(
    pl.lit('world').alias('global'),
    pl.col('year').alias('time'),
    pl.col('epov').alias('poverty_rate')
)
gbl_epov_rates

gbl_epov_rates.write_csv('./ddf/poverty_rates/ddf--datapoints--poverty_rate--by--global--time.csv')


# countries
datalist = data.partition_by(['country', 'year'], as_dict=True)

datalist[('afg', 1800)]

country_epov = list()
for k, _df in datalist.items():
    country, y = k
    country_epov.append({
        'country': country,
        'time': y,
        'poverty_rate': get_epov_rates(_df)
    })


country_epov_rates = pl.from_records(country_epov)

country_epov_rates

country_epov_rates.write_csv('./ddf/poverty_rates/ddf--datapoints--poverty_rate--by--country--time.csv')

# income groups
# 1. get historical income groups
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

income_groups = pl.DataFrame(on.reset_index()).select(
    pl.col('geo').alias('country'),
    pl.col('time').cast(pl.Int32).alias('year'),
    pl.col('level')
)

# 2. get rates
data_income_level = data.join(income_groups, on=['country', 'year'], how='left').group_by(
    pl.col(['level', 'year', 'bracket'])
).agg(
    pl.col('population').sum()
).sort(['level', 'year', 'bracket'])
data_income_level

data_income_level_list = data_income_level.partition_by(['level', 'year'], as_dict=True)

level_epov = list()
for k, _df in data_income_level_list.items():
    level, y = k
    level_epov.append({
        'level': level,
        'time': y,
        'poverty_rate': get_epov_rates(_df)
    })

level_epov_rates = pl.from_records(level_epov)

# level_epov_rates
# level_epov_rates.filter(
#     pl.col('time') == 2018
# )
# convert level to income groups
concept_id_map = {'l1': 'low_income',
                  'l2': 'lower_middle_income',
                  'l3': 'upper_middle_income',
                  'l4': 'high_income'
                 }
level_epov_rates = level_epov_rates.select(
    pl.col('level').map_dict(concept_id_map).alias('income_groups'),
    pl.col('time'),
    pl.col('poverty_rate')
)
level_epov_rates.write_csv('./ddf/poverty_rates/ddf--datapoints--poverty_rate--by--income_groups--time.csv')

# people in extreme poverty
population = pl.read_csv('../build/source/gapminder/population.csv')
population

population = population.select(
    pl.col('geo').alias('country'),
    pl.col('time'),
    pl.col('Population').alias('population')
)

pop_in_epov = country_epov_rates.join(population, on=['country', 'time'], how='left').select(
    pl.col(['country', 'time']),
    (pl.col('population') * pl.col('poverty_rate') / 100).cast(pl.Int64).alias('population_in_extreme_poverty')
)

pop_in_epov.write_csv('./ddf/poverty_rates/ddf--datapoints--population_in_extreme_poverty--by--country--time.csv')
