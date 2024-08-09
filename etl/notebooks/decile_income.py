# -*- coding: utf-8 -*-

"""Extract decile income and median income by country from bridged data
"""

import os
import os.path as osp
import sys

import numpy as np
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

# settings for display images
sns.set_context('notebook')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (7, 4)
plt.rcParams['figure.dpi'] = 144


# I will upload all files to gdrive. configure gdrive path
gdrive_path = '~/src/work/gapminder/GDrive/IncomeMountainDeciles'


# helper functions
def _f(df, **kwargs):
    return df.filter(pl.all_horizontal([(pl.col(k) == v) for k, v in kwargs.items()]))


def bracket_to_income(b, bracket_delta=0.04):   # default: 500 brackets till 8192
    return np.power(2, -7 + ((b + 1) * bracket_delta))


def income_to_bracket(i, bracket_delta=0.04):
    return int(np.ceil((np.log2(i) + 7) / bracket_delta - 1))

# NOTE: 8192 should be the right bound of bracket 499. and it's inclusive


bracket_to_income(499, 0.04)  # 2 ** 13 = 8192
bracket_to_income(0, 0.04)  # 2 ** (-7 + 0.04) = 0.00803213927075052

income_to_bracket(2 ** (-7+0.01))  # 0
income_to_bracket(2 ** (13+0.01))  # 500


# import data
data = pl.read_parquet('./bridged_shapes.parquet')


# function to export to xls file
def export_to_xls_or_csv(df: pl.DataFrame, fname: str):
    # pandas can detect if file is too large for excel.
    # so we use it
    df_pd = df.to_pandas()
    try:
        df_pd.to_excel(osp.join(gdrive_path, f'{fname}.xlsx'),
                       index=False,
                       float_format='%.4f')
    except ValueError:
        print('file too large, use csv')
        df_pd.to_csv(osp.join(gdrive_path, f'{fname}.csv'),
                     index=False,
                     float_format='%.4f')


# 1. selected years for high freq income mountain

# global shape 2021-2100
global_shape = data.filter(
    pl.col('year').is_in(range(2021, 2101))
).group_by(['year', 'bracket']).agg(
    pl.col('population').sum()
).with_columns(
    pl.lit('world').alias('global')
).sort(('year', 'bracket'))
global_shape
export_to_xls_or_csv(global_shape, 'world_brackets_1050_pop_2021-2100')

# country shape 2023
country_shape = data.filter(
    pl.col('year') == 2023
).sort(('year', 'bracket'))
export_to_xls_or_csv(country_shape, 'country_brackets_1050_pop_2023')


# - country shape in 105 brackets
# - world shape in 105 brackets
# - country shape in 50 brackets
# - world shape in 50 brackets

# function to resample data
# (taken from the scripts/income_mountain.py)
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


# create partitions
datalist = data.partition_by(['country', 'year'])

# country in 105
country105 = map(lambda x: resample_country(x, 105, 10), datalist)
country105 = pl.concat(country105)
country105
export_to_xls_or_csv(country105, "country_brackets_105_pop")

# world in 105
world105 = country105.group_by(['year', 'bracket']).agg(
    pl.col('population').sum()
).select(
    pl.lit('world').alias('global'),
    pl.all()
).sort(['year', 'bracket'])
world105
export_to_xls_or_csv(world105, "world_brackets_105_pop")


# country in 50
country50 = country105.filter(
    pl.col('bracket') < 50
)
country50
export_to_xls_or_csv(country50, "country_brackets_50_pop")

# world in 50
world50 = country50.group_by(['year', 'bracket']).agg(
    pl.col('population').sum()
).select(
    pl.lit('world').alias('global'),
    pl.all()
).sort(['year', 'bracket'])
world50
export_to_xls_or_csv(world50, "world_brackets_50_pop")

# 2. decile/ventile indicators

# precalculate the mean income of each bracket
bstart = bracket_to_income(np.arange(-1, 1199))
bend = bracket_to_income(np.arange(0, 1200))
bmean = np.sqrt(bstart * bend)
bmean = pl.Series(bmean)
len(bmean)
bmean[1192:].sum()


# get richest richest income
# df = _f(data, country='chn', year=2025)
# top_pop = (df.select('population').sum() / 1000).item()
# top_pop

def get_richest_income(ser,  # input: population count by bracket
                       splitnum: int,
                       kw: str,
                       cut=False):
    cumlpop = 0
    cumlmon = 0
    incomes = []
    tp = ser['population'].sum()
    tp1 = tp / splitnum
    # print(tp1)

    for rec in ser.to_dicts():
        i, bp = rec['bracket'], rec['population']
        if cut and i > cut:
            break
        # find the bracket where we have all but one brackets of population
        target_pop = int((splitnum - 1) * tp1)
        if cumlpop + bp < target_pop:
            cumlpop = cumlpop + bp
            cumlmon = cumlmon + (bmean[i] * bp)
        else:
            pop_this_decile = target_pop - cumlpop
            pop_next_decile = bp - pop_this_decile
            # print(
            #     "cut at {}, left pop {}, right pop {}".format(i,
            #                                                   cumlpop + pop_this_decile,
            #                                                   tp - (cumlpop + pop_this_decile)))

            money_this_decile = cumlmon + (bmean[i] * pop_this_decile)
            incomes.append(money_this_decile / target_pop)

            # now reset cumlpop and cumlmon for next decile
            cumlmon = bmean[i] * pop_next_decile
            cumlpop = pop_next_decile
            target_pop = tp - (cumlpop + pop_this_decile)
    # append the final group if it's not there
    if len(incomes) != 2:
        incomes.append(cumlmon / cumlpop)
    res = {'bracket': kw, 'income': incomes[1]}
    return pl.DataFrame(res)


# get_richest_income(df, 100000, 't_0_001')
data_part = data.partition_by(['country', 'year'], as_dict=True)


def extract_richest(func, input_data):
    res = []
    for k, v in input_data.items():
        country, year = k
        res.append(func(v, country, year))
    res = pl.concat(res)
    return res


def get_richest_1(ser, country, year, cut=False):  # divide income mountain into 1000 parts and get richest
    return get_richest_income(
        ser.select(['bracket', 'population']), 1000, 't_0_001', cut
    ).with_columns(
        pl.lit(country).alias('country'),
        pl.lit(year).alias('year')
    )


def get_richest_2(ser, country, year, cut=False):  # divide income mountain into 100000 parts and get richest
    return get_richest_income(
        ser.select(['bracket', 'population']), 100000, 't_0_00001', cut
    ).with_columns(
        pl.lit(country).alias('country'),
        pl.lit(year).alias('year')
    )


# df = _f(data, country='moz', year=2025)
# get_richest_1(df, 'moz', 2025)

rich1 = extract_richest(get_richest_1, data_part)
rich2 = extract_richest(get_richest_2, data_part)

rich1
rich2

rich_hhinc = pl.concat([rich1, rich2])

rich_hhinc = rich_hhinc.select(['country', 'year', 'bracket', 'income'])
rich_hhinc
export_to_xls_or_csv(rich_hhinc, 'richest_hhinc')


def get_split_income(ser,  # input: population count by bracket
                     splitnum: int,
                     kw: str,
                     cut=False):
    cumlpop = 0
    cumlmon = 0
    incomes = []
    tp1 = ser['population'].sum() / splitnum
    # print(tp1)

    for rec in ser.to_dicts():
        i, bp = rec['bracket'], rec['population']
        if cut and i > cut:
            break
        if cumlpop + bp < tp1:
            cumlpop = cumlpop + bp
            cumlmon = cumlmon + (bmean[i] * bp)
        else:
            pop_this_decile = tp1 - cumlpop
            pop_next_decile = bp - pop_this_decile
            # print("decile cut at {}, pop {}".format(i, cumlpop + pop_this_decile))

            money_this_decile = cumlmon + (bmean[i] * pop_this_decile)
            incomes.append(money_this_decile / tp1)

            # now reset cumlpop and cumlmon for next decile
            cumlmon = bmean[i] * pop_next_decile
            cumlpop = pop_next_decile

    # there might be floating point precision issue, the last group is not added.
    # we need to add it manually.
    if len(incomes) != splitnum:
        # print("decile cut at {}, pop {}".format(i, cumlpop))
        incomes.append(cumlmon / cumlpop)

    res = pl.Series(incomes)
    maxlength = len(str(splitnum))
    tmpl = "{{}}{{:>0{}d}}".format(maxlength)
    idxs = map(lambda x: tmpl.format(kw[0], x+1),
               range(splitnum))
    res_df = pl.DataFrame(
        dict([
            (kw, idxs),
            ('income', res)
        ])
    )
    return res_df


def get_decile_income(ser, country, year, cut=False):
    return get_split_income(
        ser.select(['bracket', 'population']), 10, 'decile', cut
    ).with_columns(
        pl.lit(country).alias('country'),
        pl.lit(year).alias('year')
    )


def get_ventile_income(ser, country, year, cut=False):
    return get_split_income(
        ser.select(['bracket', 'population']), 20, 'ventile', cut
    ).with_columns(
        pl.lit(country).alias('country'),
        pl.lit(year).alias('year')
    )


def get_centile_income(ser, country, year, cut=False):
    return get_split_income(
        ser.select(['bracket', 'population']), 100, 'centile', cut
    ).with_columns(
        pl.lit(country).alias('country'),
        pl.lit(year).alias('year')
    )


def get_quintile_income(ser, country, year, cut=False):
    return get_split_income(
        ser.select(['bracket', 'population']), 5, 'quintile', cut
    ).with_columns(
        pl.lit(country).alias('country'),
        pl.lit(year).alias('year')
    )

# df = _f(data, country='chn', year=2025)
# get_quintile_income(df)


data_part = data.partition_by(['country', 'year'], as_dict=True)
# len(data_part) * 1.6 / 1000  # estimeate time for running all computation, 90 seconds
# data_part[('afg', 1800)]


def extract_income_rank(func, input_data):
    res = []
    for k, v in input_data.items():
        country, year = k
        res.append(func(v, country, year))
    res = pl.concat(res)
    return res


decile_income = extract_income_rank(get_decile_income, data_part)
ventile_income = extract_income_rank(get_ventile_income, data_part)
quintile_income = extract_income_rank(get_quintile_income, data_part)
# _f(ventile_income, country='moz', year=2020)
decile_income
# centile only need for 2022
data_part_22 = _f(data, year=2022).partition_by(['country', 'year'], as_dict=True)
centile_income_22 = extract_income_rank(get_centile_income, data_part_22)
# centile_income_22

# export to file
# os.makedirs('other/')
export_to_xls_or_csv(decile_income.select(['country', 'year', 'decile', 'income']),
                     "decile_hhinc")
export_to_xls_or_csv(ventile_income.select(['country', 'year', 'ventile', 'income']),
                     "ventile_hhinc")
export_to_xls_or_csv(quintile_income.select(['country', 'year', 'quintile', 'income']),
                     "quintile_hhinc")
export_to_xls_or_csv(centile_income_22.select(['country', 'year', 'centile', 'income']),
                     "centile_hhinc_22")

# centile for 1940-2023
data_part_4023 = data.filter(
    pl.col('year').is_in(range(1940, 2024))
).partition_by(['country', 'year'], as_dict=True)

centile_income_4023 = extract_income_rank(get_centile_income, data_part_4023)
# centile_income_4023

res4023 = centile_income_4023.pivot(values='income', index=['country', 'centile'], columns='year')
res4023
export_to_xls_or_csv(res4023, "centile_mean_income_4023")

# centile for 2024-2100
data_part_2400 = data.filter(
    pl.col('year').is_in(range(2024, 2101))
).partition_by(['country', 'year'], as_dict=True)

centile_income_2400 = extract_income_rank(get_centile_income, data_part_2400)
# centile_income_2400

res2400 = centile_income_2400.pivot(values='income', index=['country', 'centile'], columns='year')
export_to_xls_or_csv(res2400, "centile_mean_income_2400")


# Compute the GDP based version.
meaninc = pl.read_csv('../build/source/gapminder/mean_income.csv')
meaninc = meaninc.select(
    pl.col('geo').alias('country'),
    pl.col('time').alias('year'),
    (pl.col('Average daily income per capita').alias('mhhinc'))
)
meaninc

gdppc = pl.read_csv('../build/source/gapminder/gdp.csv')
gdppc = gdppc.select(
    pl.col('geo').alias('country'),
    pl.col('time').alias('year'),
    (pl.col('Income per person') / 365.).alias('gdppc')
)
gdppc

mean_income = meaninc.join(gdppc, on=['country', 'year'], how='outer')  # there are some nulls
mean_income = mean_income.drop_nulls()
mean_income

mean_income = mean_income.with_columns(
    (pl.col('gdppc') / pl.col('mhhinc')).alias('ratio')
)

ratio = mean_income.select(['country', 'year', 'ratio']).with_columns(
    pl.col('year').cast(pl.Int32)
)

ratio


# moving the shape to centre on gdppc
# compare different ways:
# 1. convert bracket number to income, then  multiply the ratio and convert back
# [NOT WORK] 2. multiple ratio directly with bracket number
# 3. multiple ratio with decile incomes

# and experiment shows 1 is same as 3. so we should use 3

quintile_gdp = quintile_income.join(ratio, on=['country', 'year'], how='inner').with_columns(
    pl.col('income') * pl.col('ratio')
).select(pl.exclude('ratio'))
ventile_gdp = ventile_income.join(ratio, on=['country', 'year'], how='inner').with_columns(
    pl.col('income') * pl.col('ratio')
).select(pl.exclude('ratio'))
decile_gdp = decile_income.join(ratio, on=['country', 'year'], how='inner').with_columns(
    pl.col('income') * pl.col('ratio')
).select(pl.exclude('ratio'))
centile_gdp_22 = centile_income_22.join(ratio, on=['country', 'year'], how='inner').with_columns(
    pl.col('income') * pl.col('ratio')
).select(pl.exclude('ratio'))

rich_gdp = rich_hhinc.join(ratio, on=['country', 'year'], how='inner').with_columns(
    pl.col('income') * pl.col('ratio')
).select(pl.exclude('ratio'))

# centile_gdp_22
# quintile_gdp
decile_gdp

# export to file
export_to_xls_or_csv(
    decile_gdp.select(['country', 'year', 'decile', 'income']), "decile_gdp")
export_to_xls_or_csv(
    ventile_gdp.select(['country', 'year', 'ventile', 'income']), "ventile_gdp")
export_to_xls_or_csv(
    quintile_gdp.select(['country', 'year', 'quintile', 'income']), "quintile_gdp")
export_to_xls_or_csv(
    centile_gdp_22.select(['country', 'year', 'centile', 'income']), "centile_gdp_22")
export_to_xls_or_csv(
    rich_gdp, "richest_gdp")


# Median income

# calculate the population percentage
data_pct = data.with_columns(
    (pl.col('population') / pl.col('population').sum()).over(['country', 'year'])
)


def get_median_income(df):
    dfc = df.with_columns(
        (pl.col('population').cumsum()).alias('cumsum')
    )
    b0 = dfc.filter(
        pl.col('cumsum') <= 0.5
    )['bracket'][-1]

    b1 = dfc.filter(
        pl.col('cumsum') >= 0.5
    )['bracket'][0]

    if b0 == b1:  # then we have a group ends in median income exactly.
        median = bend[b0]
    else:
        # the last cumlative population before 0.5
        c0 = _f(dfc, bracket=b0)['cumsum'].item()
        # the bracket population of next group
        p1 = _f(dfc, bracket=b0)['population'].item()

        poor = (0.5 - c0) / p1
        imedian = b0 + (b1 - b0) * poor
        median = bracket_to_income(imedian)

    return median


# df = _f(data_pct, country='afg', year=1800)
# get_median_income(df)


# calculate for all country year
median_income = []
for k, v in data_pct.partition_by(['country', 'year'], as_dict=True).items():
    country, year = k
    m = get_median_income(v)
    median_income.append(
        v.select(
            pl.col(['country', 'year']).first(),
            pl.lit(m).alias('median_income')
        )
    )
median_income = pl.concat(median_income)
median_income

# gdppc version
median_gdppc = (median_income
                .join(ratio, on=['country', 'year'], how='inner')
                .select(
                    pl.col(['country', 'year']),
                    (pl.col('median_income') * pl.col('ratio')).alias('median_gdppc')
                )
            )
median_gdppc
median_gdppc.drop_nulls()  # no nulls here.


# export to file
export_to_xls_or_csv(median_income, "median_income")
export_to_xls_or_csv(median_gdppc, "median_gdp")


# Then, copy the extreme poverty rates from ddf
epov_rate = pl.read_csv('../../poverty_rates/ddf--datapoints--poverty_rate--by--country--time.csv')
export_to_xls_or_csv(epov_rate, "extreme_poverty_rate")

epov_pop = pl.read_csv('../../poverty_rates/ddf--datapoints--population_in_extreme_poverty--by--country--time.csv')
export_to_xls_or_csv(epov_pop, "people_in_extreme_poverty")

epov_rate_gbl = pl.read_csv('../../poverty_rates/ddf--datapoints--poverty_rate--by--global--time.csv')
export_to_xls_or_csv(epov_rate_gbl, "extreme_poverty_rate_global")

epov_rate_level = pl.read_csv('../../poverty_rates/ddf--datapoints--poverty_rate--by--income_groups--time.csv')
export_to_xls_or_csv(epov_rate_level, "extreme_poverty_rate_income_groups")


# NOTE: if use rclone to upload to gdrive
# use this command to sync (make remote the sams as local):
# rclone sync --drive-import-formats "xlsx" IncomeMountainDeciles/ gdrive:/IncomeMountainDeciles/
# use this command to copy (just copy files to remote):
# rclone copy --drive-import-formats "xlsx" IncomeMountainDeciles/export/ gdrive:/IncomeMountainDeciles/
# only import xlsx to google spreadsheet. csv for large files and no need to import

# TODO:
# - sanity checking?
# plt.plot(df['bracket'], df['population'])
# plt.show()

# data_pct
# data
