# -*- coding: utf-8 -*-

"""A script to produce all varients of income mountain to csv file.
"""

import numpy as np
import polars as pl
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
sns.set_context('notebook')


def _f(df, **kwargs):
    return df.filter(pl.all([(pl.col(k) == v) for k, v in kwargs.items()]))


data = pickle.load(open('./bridged_shapes.pkl', 'rb'))

# data


# data['bracket'].max()


# data.filter(pl.col('bracket') > 900)['country'].unique()


# df = _f(data, country='idn', year=2069)

# plt.plot(df['bracket'], df['population'])
# plt.show()

# df.with_columns(
#     (pl.col('bracket') / 10).cast(pl.Int32).alias('bnew')
# ).groupby(['country', 'year', 'bnew'], maintain_order=True).agg(
#     pl.col('population').sum()
# ).select(
#     pl.col(['country', 'year']),
#     pl.col('bnew').alias('bracket'),
#     pl.col('population')
# )


datalist = data.partition_by(['country', 'year'])


def resample_country(df, rmax=None, scale=10):
    country = df['country'][0]
    year = df['year'][0]
    df_ = df.with_columns(
        (pl.col('bracket') / scale).cast(pl.Int32).alias('bnew')
    ).groupby(['bnew'], maintain_order=True).agg(
        pl.col('population').sum()
    ).select(
        pl.col('bnew').alias('bracket'),
        pl.col('population')
    )
    if rmax is None:
        full_range = pl.Series('bracket', range(df_['bracket'].max() + 1), dtype=pl.Int32).to_frame()
    else:
        full_range = pl.Series('bracket', range(rmax + 1), dtype=pl.Int32).to_frame()

    return df_.join(
        full_range, on='bracket', how='outer'
    ).with_columns(
        pl.lit(country).alias('country'),
        pl.lit(year).alias('year'),
        pl.col('population').fill_null(0)
    ).select(['country', 'year', 'bracket', 'population'])



res = map(resample_country, datalist)

res = pl.concat(res)

res.select('bracket').max()

res.write_csv('./country_shape_105.csv')

res_gbl = res.groupby(['year', 'bracket'], maintain_order=True).agg(
    pl.col('population').sum()
)
res_gbl = res_gbl.select(
    pl.lit('world').alias('global'),
    pl.all()
)

res_gbl

res_gbl.write_csv('./world_shape_105.csv')

res_pivot_gbl = res_gbl.pivot(values='population', index=['global', 'year'], columns='bracket', aggregate_function=None)

out_gbl = res_pivot_gbl.select(
    pl.col('global'),
    pl.col('year').alias('time'),
    pl.struct(pl.col(map(str, (range(0, 105))))).apply(join_str).alias('income_mountain_105bracket_shape_for_log')
)

out_gbl

out_gbl.write_csv('ddf/income_mountain/ddf--datapoints--income_mountain_105bracket_shape_for_log--by--global--time.csv')

res_pivot = res.pivot(values='population', index=['country', 'year'], columns='bracket', aggregate_function=None)

res_pivot = res_pivot.fill_null(0)


def join_str(d):
    return ','.join(map(str, d.values()))


out = res_pivot.select(
    pl.col('country'),
    pl.col('year').alias('time'),
    pl.struct(pl.col(map(str, (range(0, 105))))).apply(join_str).alias('income_mountain_105bracket_shape_for_log')
)

out

out.write_csv('ddf/income_mountain/ddf--datapoints--income_mountain_105bracket_shape_for_log--by--country--time.csv')


# df = _f(res, country='usa', year=2069)
# plt.plot(df['bracket'], df['population'])
# plt.show()

# produce global shapes in very small brackets

res_recent = data.filter(pl.col('year').is_in([2021, 2022, 2023]))

res_recent

res_gbl = res_recent.groupby(['year', 'bracket'], maintain_order=True).agg(
    pl.col('population').sum()
)

res_gbl = res_gbl.select(
    pl.lit('world').alias('global'),
    pl.col('year').alias('time'),
    pl.col('bracket'),
    pl.col('population')
)

res_gbl

dlist = res_gbl.partition_by(['global', 'time'])


def resample_global(df, rmax=None, scale=10):
    year = df['time'][0]
    df_ = df.with_columns(
        (pl.col('bracket') / scale).cast(pl.Int32).alias('bnew')
    ).groupby(['bnew'], maintain_order=True).agg(
        pl.col('population').sum()
    ).select(
        pl.col('bnew').alias('bracket'),
        pl.col('population')
    )
    if rmax is None:
        full_range = pl.Series('bracket', range(df_['bracket'].max() + 1), dtype=pl.Int32).to_frame()
    else:
        full_range = pl.Series('bracket', range(rmax + 1), dtype=pl.Int32).to_frame()

    return df_.join(
        full_range, on='bracket', how='outer'
    ).with_columns(
        pl.lit('world').alias('global'),
        pl.lit(year).alias('time'),
        pl.col('population').fill_null(0)
    ).select(['global', 'time', 'bracket', 'population'])


res_gbl = map(lambda x: resample_global(x, scale=1), dlist)

res_gbl = pl.concat(res_gbl)

res_gbl

res_gbl.write_csv('./global_shapes_2021_2023.csv')

res_gbl_p = res_gbl.pivot(values='population', index=['global', 'time'], columns='bracket', aggregate_function=None)

res_gbl_p = res_gbl_p.fill_null(0)

res_gbl_p

out2 = res_gbl_p.select(
    pl.col('global'),
    pl.col('time'),
    pl.struct(pl.col(map(str, (range(0, 861))))).apply(join_str).alias('income_mountain_1050bracket_shape_for_log')
)

out2

out2.write_csv('ddf/income_mountain/ddf--datapoints--income_mountain_1050bracket_shape_for_log--by--global--time.csv')
