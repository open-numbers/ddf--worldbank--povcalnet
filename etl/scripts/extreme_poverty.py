"""calculate the extreme poverty rate
"""
import os
import numpy as np
import polars as pl
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
