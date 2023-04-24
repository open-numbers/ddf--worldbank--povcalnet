# -*- coding: utf-8 -*-

"""Extract decile income and median income by country from bridged data
"""

import os
import sys

import numpy as np
import polars as pl
import pickle

import etllib

import matplotlib.pyplot as plt
import seaborn as sns

# settings for display images
sns.set_context('notebook')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (7, 4)
plt.rcParams['figure.dpi'] = 144


def _f(df, **kwargs):
    return df.filter(pl.all([(pl.col(k) == v) for k, v in kwargs.items()]))

# main

# import data
data = pickle.load(open('./bridged_shapes.pkl', 'rb'))


def bracket_to_income(b, bracket_delta=0.04):   # default: 500 brackets till 8192
    return np.power(2, -7 + ((b + 1) * bracket_delta))


def income_to_bracket(i, bracket_delta=0.04):
    return int(np.ceil((np.log2(i) + 7) / bracket_delta - 1))

# NOTE: 8192 should be the right bound of bracket 499. and it's inclusive


bracket_to_income(499, 0.04)  # 2 ** 13 = 8192
bracket_to_income(0, 0.04)  # 2 ** (-7 + 0.04) = 0.00803213927075052

income_to_bracket(2 ** (-7+0.01))  # 0
income_to_bracket(2 ** (13+0.01))  # 500


# precalculate the mean income of each bracket
bstart = bracket_to_income(np.arange(-1, 1199))
bend = bracket_to_income(np.arange(0, 1200))
bmean = np.sqrt(bstart * bend)
bmean = pl.Series(bmean)
bmean
bmean[1199]

# let's try 2022 first
df22usa = _f(data, country='usa', year=2022).select(['bracket', 'population'])
df22usa


def get_split_income(ser,  # input: population count by bracket
                     splitnum: int,
                     kw: str,
                     cut=False):
    cumlpop = 0
    cumlmon = 0
    incomes = []
    tp1 = ser['population'].sum() / splitnum

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
        ser.select(['bracket', 'population']), 5, 'qunitile', cut
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

# centile only need for 2022
data_part_22 = _f(data, year=2022).partition_by(['country', 'year'], as_dict=True)
centile_income_22 = extract_income_rank(get_centile_income, data_part_22)
# centile_income_22


# Compute the GDP based version.

# FIXME: use latest mean income and GDP
meaninc = pl.read_csv('../../../ddf--gapminder--fasttrack/ddf--datapoints--mhhinc--by--country--time.csv')
meaninc = meaninc.select(
    pl.col('country'),
    pl.col('time').alias('year'),
    (pl.col('mhhinc') / 365.)
)
gdppc = pl.read_csv('../source_bak/fixtures/gdppc_latest.csv')
gdppc = gdppc.select(
    pl.col('geo').alias('country'),
    pl.col('time').alias('year'),
    (pl.col('Income per person') / 365.).alias('gdppc')
)

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
# centile_gdp_22
# quintile_gdp



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


df = _f(data_pct, country='afg', year=1800)
get_median_income(df)


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


# NEXT:
# - save those into files
# - sanity checking?


plt.plot(df['bracket'], df['population'])
plt.show()

data_pct
data
