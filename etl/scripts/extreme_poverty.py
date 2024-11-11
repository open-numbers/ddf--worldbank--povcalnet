"""calculate the extreme poverty rate
"""
import os
import numpy as np
import polars as pl
import pandas as pd
import bracketlib
from functools import lru_cache

data = pl.read_parquet('../build/bridged_shapes.parquet')
total_pop = pl.read_csv('../build/source/gapminder/population.csv')
total_pop = total_pop.select(
    pl.col('geo').alias('country'),
    pl.col('time').alias('year').cast(pl.Int32),
    pl.col('Population').alias('population')
)
gbl_total_pop = total_pop.group_by('year').agg(
    pl.col('population').sum()
)
gbl_total_pop
data = data.join(total_pop, on=['country', 'year'], how='left', suffix='_total')

# dir to store indicators
os.makedirs('./ddf/poverty_rates/', exist_ok=True)
os.makedirs('./other', exist_ok=True)

step = bracketlib.get_bracket_step(500)
step

# where is 2.15/day
upper_bracket = bracketlib.bracket_from_income(2.15, step)
lower_bracket = upper_bracket - 1

# use 10 base to avoid this number gets too big.
# the result is just about the same as using base 2
xloc = np.log10(2.15)


# a cached function to speed things up
@lru_cache()
def income_from_bracket(x):
    return np.log10(bracketlib.income_from_bracket(x, step, integer=False))


def get_epov_rates(df, xloc, lb, ub, rates=True):
    """Function to get poverty rates under a poverty line.

    This function should be used per geo.
    Parameter rates=False means return total population instead of rates.
    """
    total_pop_actual = df['population_total'][0]
    total_pop_data = df['population'].sum()
    initial_val = (total_pop_actual - total_pop_data) / total_pop_actual
    if initial_val < 0:
        initial_val = 0
    df_ = df.sort('bracket').with_columns(
        (initial_val + (pl.col('population').cum_sum() / total_pop_actual)).alias("poverty_pop")
    ).filter(
        pl.col('bracket').is_in([lb, ub])
    ).with_columns(
        # note: the log base should be the same as how xloc calculated
        pl.col('bracket').map_elements(
            income_from_bracket,
            return_dtype=pl.Float64
        )
    )
    if not rates:
        df_ = df_.with_columns(
            (pl.col('poverty_pop') * total_pop_actual).floor().cast(pl.Int64)
        )
    if df_.is_empty():
        if rates:
            return 0.0
        return 0
    else:
        x = df_['bracket'].to_numpy()
        y = df_['poverty_pop'].to_numpy()

        xnew = np.array([xloc])
        ynew = np.interp(xnew, x, y)
        if rates:
            return ynew[0] * 100
        else:
            return int(ynew[0] * 100)


# let's try global first
data_gbl = data.group_by(['year', 'bracket']).agg(pl.col('population').sum())
data_gbl
data_gbl = data_gbl.join(gbl_total_pop, on='year', how='left', suffix='_total')
data_gbl.sort('year', 'bracket')

get_epov_rates(data_gbl.filter(
    pl.col('year') == 2021
), lb=lower_bracket, ub=upper_bracket, xloc=xloc, rates=True)


# 
def get_epov_rates_for_groups(df, by, povline, rates=True):
    """a function which will partition the data into groups and process all
    groups and return the result

    when rates=False, return total population count instead.
    """
    ub = bracketlib.bracket_from_income(povline, step)
    lb = ub - 1
    xloc = np.log10(povline)

    if povline == 2.15:
        name = "extreme_poverty_rate"
    else:
        s = str(povline).replace('.', '_')
        name = f"poverty_rate_under_{s}"

    datalist = df.partition_by(by, as_dict=True)
    res = list()

    for k, _df in datalist.items():
        res_part = dict(zip(by, k))
        res_part[name] = get_epov_rates(_df,
                                        lb=lb,
                                        ub=ub,
                                        xloc=xloc,
                                        rates=rates)
        res.append(res_part)

    return pl.from_records(res)


gbl_epov_rates = get_epov_rates_for_groups(data_gbl, ["year"], 2.15).sort("year")
gbl_epov_rates = gbl_epov_rates.select(
    pl.lit('world').alias('global'),
    pl.col("year").alias("time"),
    pl.exclude("year")
)
gbl_epov_rates

gbl_epov_rates.write_csv('./ddf/poverty_rates/ddf--datapoints--poverty_rate--by--global--time.csv')

# 3.65 
gbl_epov_rates_365 = get_epov_rates_for_groups(data_gbl, ["year"], 3.65).sort("year")
gbl_epov_rates_365 = gbl_epov_rates_365.select(
    pl.lit('world').alias('global'),
    pl.col("year").alias("time"),
    pl.exclude("year")
)

# 6.85
gbl_epov_rates_685 = get_epov_rates_for_groups(data_gbl, ["year"], 6.85).sort("year")
gbl_epov_rates_685 = gbl_epov_rates_685.select(
    pl.lit('world').alias('global'),
    pl.col("year").alias("time"),
    pl.exclude("year")
)


# create a merged version
global_rates = gbl_epov_rates.join(
    gbl_epov_rates_365, on=["global", "time"], how="full", coalesce=True
).join(
    gbl_epov_rates_685, on=["global", "time"], how="full", coalesce=True
)
global_rates
global_rates.write_csv('./other/epov_by_global.csv')


# countries - 2.15

country_epov_rates = get_epov_rates_for_groups(data, ["country", "year"], 2.15)
country_epov_rates = country_epov_rates.rename({"year": "time"})
country_epov_rates

country_epov_rates.filter(
    (pl.col('country') == 'sle') & (pl.col('time') == 1981)
)
# => 64.22, correct!

country_epov_rates.filter(
    (pl.col('country') == 'nru') & (pl.col('time') == 2009)
)

country_epov_rates.write_csv('./ddf/poverty_rates/ddf--datapoints--extereme_poverty_rate--by--country--time.csv')

# countries - 3.65
country_epov_rates_365 = get_epov_rates_for_groups(data, ["country", "year"], 3.65)
country_epov_rates_365 = country_epov_rates_365.rename({"year": "time"})
country_epov_rates_365

country_epov_rates_365.write_csv('./ddf/poverty_rates/ddf--datapoints--poverty_rate_under_3_65--by--country--time.csv')

# countries - 6.85
country_epov_rates_685 = get_epov_rates_for_groups(data, ["country", "year"], 6.85)
country_epov_rates_685 = country_epov_rates_685.rename({"year": "time"})
country_epov_rates_685

country_epov_rates_685.write_csv('./ddf/poverty_rates/ddf--datapoints--poverty_rate_under_6_85--by--country--time.csv')

# create a merged version
country_rates = country_epov_rates.join(
    country_epov_rates_365, on=["country", "time"], how="full", coalesce=True
).join(
    country_epov_rates_685, on=["country", "time"], how="full", coalesce=True
)
country_rates
country_rates.write_csv('./other/epov_by_country.csv')

###############
# income groups
# 1. get historical income groups
wb_groups = pd.read_csv('../build/source/gapminder/wb_income_groups.csv')
wb_groups

wb = wb_groups[['geo', 'time', "4 income groups (WB)"]].copy()

wb.columns = ['geo', 'time', 'level']

on_income = pd.read_csv('../build/source/gapminder/on_income_groups.csv')
on_income = on_income[['geo', 'time', 'Income levels']].dropna()
on_income.columns = ['geo', 'time', 'level']

on_map = {'Level 1': 'l1',
          'Level 2': 'l2',
          'Level 3': 'l3',
          'Level 4': 'l4',
          'Level 5': 'l4'   # map level 5 to high income
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

# convert level to income groups
concept_id_map = {'l1': 'low_income',
                  'l2': 'lower_middle_income',
                  'l3': 'upper_middle_income',
                  'l4': 'high_income'
                 }

income_groups = pl.DataFrame(on.reset_index()).select(
    pl.col('geo').alias('country'),
    pl.col('time').cast(pl.Int32).alias('year'),
    pl.col('level').replace_strict(concept_id_map)
)
income_groups

income_3groups = income_groups.select(
    pl.col(['country', 'year']),
    pl.col('level').replace({'upper_middle_income': 'middle_income', 'lower_middle_income': 'middle_income'})
)
income_3groups

# 2. get rates
data_income_level = data.join(income_groups, on=['country', 'year'], how='left').group_by(
    pl.col(['level', 'year', 'bracket'])
).agg(
    pl.col('population').sum()
).sort(['level', 'year', 'bracket'])
data_income_level

data_income_3level = data.join(income_3groups, on=['country', 'year'], how='left').group_by(
    pl.col(['level', 'year', 'bracket'])
).agg(
    pl.col('population').sum()
).sort(['level', 'year', 'bracket'])
data_income_3level


# 2.15
# level_epov_rates = get_epov_rates_for_groups(data_income_level, ["level", "year"], 2.15)
# level_epov_rates
# level_epov_rates.filter(pl.col("year").is_between(2020, 2021))

# level_epov_rates = level_epov_rates.select(
#     pl.col('level').replace_strict(concept_id_map).alias('income_groups'),
#     pl.col('year').alias('time'),
#     pl.col('poverty_rate'),
# )
# level_epov_rates.write_csv('./ddf/poverty_rates/ddf--datapoints--poverty_rate--by--income_groups--time.csv')

# 
def get_level_epov_rates(povline, use_3levels=False):
    if use_3levels:
        df = data_income_3level
    else:
        df = data_income_level
    if povline == 2.15:
        name = "extreme_poverty_rate"
    else:
        s = str(povline).replace('.', '_')
        name = f"poverty_rate_under_{s}"
    rates = get_epov_rates_for_groups(df, ["level", "year"], povline)
    return rates.select(
        pl.col('level').alias('income_group'),
        pl.col('year').alias('time'),
        pl.col(name)
    )


level_epov_215 = get_level_epov_rates(2.15)
level_epov_215

level_epov_215.write_csv('./ddf/poverty_rates/ddf--datapoints--poverty_rate--by--income_groups--time.csv')

level_epov_215_3lvl = get_level_epov_rates(2.15, use_3levels=True)
level_epov_215_3lvl

level_epov_215_3lvl.write_csv('./ddf/poverty_rates/ddf--datapoints--poverty_rate--by--income_3groups--time.csv')


# 3.65
level_epov_365 = get_level_epov_rates(3.65)
level_epov_365.write_csv('./ddf/poverty_rates/ddf--datapoints--poverty_rate_365--by--income_groups--time.csv')

level_epov_365_3lvl = get_level_epov_rates(3.65, use_3levels=True)
level_epov_365_3lvl.write_csv('./ddf/poverty_rates/ddf--datapoints--poverty_rate_365--by--income_3groups--time.csv')

# 6.85
level_epov_685 = get_level_epov_rates(6.85)
level_epov_685.write_csv('./ddf/poverty_rates/ddf--datapoints--poverty_rate_685--by--income_groups--time.csv')

level_epov_685_3lvl = get_level_epov_rates(6.85, use_3levels=True)
level_epov_685_3lvl.write_csv('./ddf/poverty_rates/ddf--datapoints--poverty_rate_685--by--income_3groups--time.csv')

# create a merged version
level_rates = level_epov_215.join(
    level_epov_365, on=["income_group", "time"], how="full", coalesce=True
).join(
    level_epov_685, on=["income_group", "time"], how="full", coalesce=True
)
level_rates
level_rates.write_csv('./other/epov_by_4levels.csv')

level_rates_3lvl = level_epov_215_3lvl.join(
    level_epov_365_3lvl, on=["income_group", "time"], how="full", coalesce=True
).join(
    level_epov_685_3lvl, on=["income_group", "time"], how="full", coalesce=True
)
level_rates_3lvl
level_rates_3lvl.write_csv('./other/epov_by_3levels.csv')


#########################
# rates by world_4regions
countries = pd.read_csv('../../ddf--entities--geo--country.csv')
countries_regions_mapping = countries.set_index('country')['world_4region'].dropna().to_dict()

data_world_4region = data.select(
    pl.col('country').replace_strict(countries_regions_mapping).alias('world_4region'),
    pl.col('year'),
    pl.col('bracket'),
    pl.col('population')
).group_by("world_4region", "year", "bracket").agg(pl.col("population").sum())

# 2.15
region_epov = get_epov_rates_for_groups(data_world_4region, ["world_4region", "year"], 2.15)
region_epov
# region_epov.write_csv('./ddf/poverty_rates/ddf--datapoints--poverty_rate--by--world_4region--year.csv')

region_epov_365 = get_epov_rates_for_groups(data_world_4region, ["world_4region", "year"], 3.65)
region_epov_365

region_epov_685 = get_epov_rates_for_groups(data_world_4region, ["world_4region", "year"], 6.85)
region_epov_685

# merge 
region_rates = region_epov.join(
    region_epov_365, on=["world_4region", "year"], how="full", coalesce=True
).join(
    region_epov_685, on=["world_4region", "year"], how="full", coalesce=True
)
region_rates
region_rates.write_csv('./other/epov_by_4region.csv')


###########################
# people in poverty
population = pl.read_csv('../build/source/gapminder/population.csv')
population

epov_under_2 = get_epov_rates_for_groups(data, ["country", "year"], 2)

population = population.select(
    pl.col('geo').alias('country'),
    pl.col('time').alias('year'),
    pl.col('Population').alias('population')
)

# population under $2
pop_in_epov_2 = epov_under_2.join(population, on=['country', 'year'], how='left').select(
    pl.col(['country', 'year']),
    (pl.col('population') * pl.col('poverty_rate_under_2') / 100).cast(pl.Int64).alias('population_under_2')
)
pop_in_epov_2


# population above $200
epov_under_200 = get_epov_rates_for_groups(data, ["country", "year"], 200)
epov_under_200

pop_above_200 = epov_under_200.join(population, on=['country', 'year'], how='left').select(
    pl.col(['country', 'year']),
    (pl.col('population') * (1 - pl.col('poverty_rate_under_200') / 100)).cast(pl.Int64).alias('population_above_200')
)
pop_above_200

# merge
pop = pop_in_epov_2.join(pop_above_200, on=["country", "year"], how="full", coalesce=True)
pop

pop.write_csv('./other/pop_2_200.csv')

# aggregate for global
pop_global = pop.group_by("year").agg(
    pl.col("population_under_2").sum(),
    pl.col("population_above_200").sum()
).select(
    pl.lit("world").alias("global"),
    pl.col("*")
).sort("year")
pop_global

pop_global.write_csv("./other/pop_2_200_global.csv")
