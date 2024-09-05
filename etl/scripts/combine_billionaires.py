# -*- coding: utf-8 -*-


"""Calculate the bridges between povcalnet and billionaires dataset.

We say left shape is from povcalnet;
Right shape is from billionaires dataset.

There are 2 cases:
- CASE 1: we have both left ard right shape.
- CASE 2: we only have left shape.

We will first calculate the shape for CASE1,
then collect some shape parameters and use these parameters
to calculate the shapes for CASE2.

Check the functions below for more details.
"""

# %%
# import os
# import sys

import numpy as np
import polars as pl
import pandas as pd
# import json
# from multiprocessing import get_context
# from functools import partial

# import etllib
# import constants
# import step3

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.interpolate import PchipInterpolator

# %%
# settings for display images
sns.set_context('notebook')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (11, 6)
plt.rcParams['figure.dpi'] = 144


brackets_delta_robin = 0.04


def _f(df, **kwargs):
    return df.filter(pl.all_horizontal([(pl.col(k) == v) for k, v in kwargs.items()]))


def plot(*args, **kwargs):
    _, ax = plt.subplots(1, 1)
    for df in args:
        plt.plot(df['bracket'], df['population'])
    if kwargs.get('log', None):
        ax.set_yscale('log')
    plt.show()


def bracket_number_from_income_robin(s, integer=True):
    res = ((np.log2(s) + 7) / brackets_delta_robin)
    if integer:
        return res.astype(int)
    return res


def bracket_number_to_income_robin(n):
    res = np.power(2, (n+1) * brackets_delta_robin - 7)
    return res


def interpolate(df, extrapolate=False):
    """interpolate a series with PCHIP. See PchipInterpolator doc
    """
    # FIXME: we don't need to convert to pandas.
    ser = df.to_pandas()
    ser_ = ser.dropna()
    name = ser.name
    res = PchipInterpolator(ser_.index, ser_.values, extrapolate=extrapolate)(ser.index)
    return pl.Series(name, res)


def calculate_bridge(left, right, scale=1000):
    """input left shape and right shape, calculate the bridged shape

    also return some stats of the bridge, useful for calculate CASE2 shapes.
    """
    # We calculate 3 points:
    # 1. the beginning of the bridge, which is in the right side of left shape, where
    #    the population number drops below a threshold (right maximum * scale)
    # 2. 2 points to define the slope of right shape.
    # 2.1 x = first bracket of right shape, y = 1.1 * maximum y of right shape
    #     We believe that the poorest billionaires are under-reported
    #     so we add more people to this group
    # 2.2 x = the middle bracket of right shape, y = the mean of y values around the middle
    # Then, base on these 3 points, we interploate missing values, using PCHIP method
    # see interpolate() method.
    #
    # left part
    right_max = int(right['population'].max() * 1.0)
    thres = right_max * scale
    left_top = left[left['population'].arg_max()]
    left_top_x = left_top['bracket'].item()
    left_top_y = left_top['population'].item()
    bridge_start = left.filter(
        (pl.col('population') > thres)
    ).select(
        pl.col('bracket').last()
    ).item()
    left1 = left.filter(
        pl.col('bracket').is_between(left_top_x, bridge_start)
    ).select(['bracket', 'population'])

    # 2 right points
    right_start = right['bracket'].min()
    right_end = right['bracket'].max()
    right_span = right_end - right_start
    right1 = pl.DataFrame({
        'bracket': right_start,
        'population': right_max
    }, schema=left1.schema)

    middle = right_start + (right_span // 2)
    right_idx = pl.Series(
        'bracket',
        range(right_start, right_end+1),
        dtype=pl.Int32
    ).to_frame()
    bridge_end_y = right.join(
        right_idx, on='bracket', how='full', coalesce=True
    ).with_columns(
        pl.col('population').interpolate()
    ).filter(
        pl.col('bracket').is_in(range(middle-5, middle+5+1))
    )['population'].mean()

    right2 = pl.DataFrame({
        'bracket': middle,
        'population': int(bridge_end_y)
    }, schema=left1.schema)

    new_i = pl.Series(
        'bracket',
        range(left_top_x, middle+1),
        dtype=pl.Int32).to_frame()
    res = pl.concat([left1, right1, right2]).with_columns(
        np.log(pl.col('population'))
    )
    res = res.join(new_i, on='bracket', how='full', coalesce=True)
    res = res.with_columns(
        np.exp(pl.col('population').map_batches(interpolate))
    )
    # calculate some parameters
    if bridge_end_y == 0:
        slope = - np.log(right_max) / (middle - right_start)
    else:
        slope = (np.log(bridge_end_y) - np.log(right_max)) / (middle - right_start)

    params = {
        'slope': slope,
        'bridge_left_y': left1['population'][-1],
        'mountain_top_y': left_top_y
    }

    res = res.with_columns(
        pl.col('population').round(0).cast(pl.Int64)
    )

    bridge_shape_all = left.select(
        ['bracket', 'population']
    ).join(
        right.select(['bracket', 'population']),
        on='bracket',
        how='full',
        suffix='_right',
        coalesce=True
    ).join(
        res,
        on='bracket',
        how='full',
        suffix='_bridge',
        coalesce=True
    )
    bridge_shape_all = bridge_shape_all.select(
        pl.col('bracket'),
        pl.coalesce(
            pl.col(['population_bridge', 'population_right', 'population']), pl.lit(0)).alias('population')
    ).sort('bracket')
    # print(bridge_shape_all)
    return params, bridge_shape_all


def calculate_bridge_2(left_shape, params):
    """calculate the bridge for CASE2
    """
    # All CASE 2 bridges will be a straight line in log scale.
    # from params, we get:
    # - the bracket where the bridge should begin. measure with the
    #   distant from the maximum of the left shape
    # - the slope: the rate of change after the bridge begins
    top_y = left_shape['population'].max()
    top_y_x = left_shape['population'].arg_max()
    left_shape_ = left_shape[top_y_x:]

    thres = np.exp(np.log(top_y) * params['ptc_top'])

    if thres < 10:
        return None

    bridge_start_x = left_shape_.filter(
        pl.col('population') <= thres
    )['bracket'][0]  # should I check indexerror here?

    bridge_first_val = np.log(left_shape.filter(
        pl.col('bracket') == bridge_start_x
    )['population'].item())
    bridge_vals = [bridge_first_val]
    slope = params['slope']

    for i in range(1000):
        new_val = bridge_first_val + (i * slope)
        if new_val < 0:
            break
        bridge_vals.append(new_val)

    bridge_arr = np.array(bridge_vals)
    bridge_arr = np.round(np.exp(bridge_arr)).astype(np.int64)
    brackets = (np.arange(len(bridge_arr)) + bridge_start_x).astype(np.int32)
    bridge_shape = pl.DataFrame(
        {
            'bracket': brackets,
            'population': bridge_arr
        }
    )
    # if our bridge is narrower than the orign povcalnet, then just use povecalnet
    if bridge_shape['bracket'].max() < left_shape['bracket'].max():
        return left_shape.select(['bracket', 'population']).clone()
    else:
        left_shape1 = left_shape.filter(
            pl.col('bracket') < bridge_start_x
        ).select(['bracket', 'population'])
        bridge_shape_all = pl.concat([left_shape1, bridge_shape])
        return bridge_shape_all.sort('bracket')


def plot_shape(df, **kwargs):
    df_ = df.filter(pl.col('population') > 0)
    plt.plot(df_['bracket'], df_['population'], **kwargs)


def make_checking_plots(povcalnet, billy_pop, all_shapes):

    # 1. show a global chart
    gleft = povcalnet.group_by(['year', 'bracket']).agg(
        pl.col('population').sum()
    )
    gbridge = all_shapes.group_by(['year', 'bracket']).agg(
        pl.col('population').sum()
    )
    gright = billy_pop.group_by(['year', 'bracket']).agg(
        pl.col('population').sum()
    )

    ts = [2022, 2023, 2060, 2100]
    for t in ts:
        left = _f(gleft, year=t).sort('bracket')
        right = _f(gright, year=t).sort('bracket')
        # right = reindex_shape(right).with_columns(
        #     pl.col('year').fill_null(pl.lit(t)),
        #     pl.col('population').fill_null(pl.lit(1))
        # )
        bridge = _f(gbridge, year=t).sort('bracket')
        diff = (bridge['population'].sum() - left['population'].sum()) / left['population'].sum() * 100
        print(f"{t}: population added {diff:.4f}%")
        plt.figure()
        _, ax = plt.subplots(1, 1)
        plot_shape(bridge[300:], label='bridged')
        plot_shape(left[300:], label='povcalnet')
        plot_shape(right, alpha=.4, label='billy')
        ax.set_yscale('log')
        # ax.set_ylim((0, 1000))
        plt.title("global, " + str(t))
        plt.legend()
        plt.savefig(f"./bridge_{t}.png")

    # 2. show a chart for CASE2 countries
    df1 = _f(povcalnet, country='vnm', year=2030)
    df2 = _f(all_shapes, country='vnm', year=2030)
    plt.figure()
    plot_shape(df1[350:], label='povcalnet')
    plot_shape(df2[350:], label='bridge')
    plt.title('vnm, 2030')
    plt.legend()
    plt.savefig('./bridge_vnm_2030.png')


if __name__ == '__main__':
    # load data
    povcalnet = pl.read_parquet('../build/population_500plus.parquet')

    # FIXME: put billionaires data into source dir or download from url
    billy = pl.read_csv('../../../ddf--gapminder--forbes_billionaires/ddf--datapoints--daily_income--by--person--time.csv')
    billy_worth = pl.read_csv('../../../ddf--gapminder--forbes_billionaires/ddf--datapoints--worth--by--person--time.csv')
    billy = billy.with_columns(
        pl.col('time').cast(pl.Int32),
        pl.col('daily_income').cast(pl.Float64)
    )
    billy_worth = billy_worth.with_columns(
        pl.col('time').cast(pl.Int32),
        pl.col('worth').cast(pl.Float64)
    )
    billy_ent = pl.read_csv('../../../ddf--gapminder--forbes_billionaires/ddf--entities--person.csv', infer_schema_length=None)
    billy_ent = billy_ent.select(['person', 'countries']).drop_nulls()

    billy_country_map = dict([(d['person'], d['countries']) for d in billy_ent.to_dicts()])
    # billy_country_map['elon_musk'] => 'usa'

    # calculate G, the average growth rate of income, using average net worth growth.
    rich2022 = _f(billy_worth, time=2022).sort(['worth'], descending=True)[:300]
    rich2012 = _f(billy_worth, time=2012).sort(['worth'], descending=True)[:300]
    df_ann = pl.DataFrame({'2012': rich2012['worth'], '2022': rich2022['worth']})
    df_ann = df_ann.with_columns(
        (np.log(pl.col('2022') / pl.col('2012')) / 10).alias('growth')
    )
    df_ann
    G = df_ann['growth'].mean()
    print("annual income growth rate: ", G)  # 0.05746843339165423

    # Question: what's the group for elon, 2022?
    # billy.with_columns(
    #     pl.col('daily_income').apply(bracket_number_from_income_robin).alias('bracket')
    # ).filter(pl.col('person') == "elon_musk")[-1]
    # Out:
    # ┌───────────┬──────┬──────────────┬─────────┐
    # │ person    ┆ time ┆ daily_income ┆ bracket │
    # │ ---       ┆ ---  ┆ ---          ┆ ---     │
    # │ str       ┆ i32  ┆ f64          ┆ i64     │
    # ╞═══════════╪══════╪══════════════╪═════════╡
    # │ elon_musk ┆ 2022 ┆ 1.7028e8     ┆ 858     │
    # └───────────┴──────┴──────────────┴─────────┘

    # get bracket for 2023-2100
    # use top 2000 for 2022 as base
    top2022 = _f(billy, time=2022).sort('daily_income', descending=True)[:2000]
    top2100 = [top2022]
    for i in range(2023, 2101):
        _df = top2100[-1].with_columns(
            pl.lit(i).alias('time'),
            pl.col('daily_income') * (1 + G),
        )
        top2100.append(_df)
    top2100 = pl.concat(top2100[1:])
    billy_full = pl.concat([billy, top2100])

    def _get_geo(x):
        res = billy_country_map[x]
        if ';' in res:  # there are people wilt multiple countries, just use the first one.
            return res.split(';')[0]
        return res

    # convert name to geo, get total count
    billy_pop = billy_full.with_columns(
        pl.col('person').map_elements(_get_geo, return_dtype=pl.Utf8).alias('country'),
        (pl.col('daily_income')
         .map_elements(bracket_number_from_income_robin, return_dtype=pl.Int32)
         .alias('bracket'))
    ).group_by(['country', 'time', 'bracket']).agg(
        pl.col('daily_income').count().alias('population')
    ).sort(['country', 'time', 'bracket'])
    billy_pop.columns = ['country', 'year', 'bracket', 'population']

    billy_pop = billy_pop.with_columns(
        pl.col('population').cast(pl.Int64)
    )

    # try to plot elon's income over time
    # df = _f(billy_full, person='bill_gates')
    # plt.plot(df['time'], df['daily_income'].apply(bracket_number_from_income_robin))
    # plt.show()

    # try the bridge method
    # geo, t = 'ind', 2020
    # left = _f(povcalnet, country=geo, year=t)
    # right = _f(billy_pop, country=geo, year=t)
    # params, res = calculate_bridge(left, right, 1000)
    # plot(left, right, res, log=True)
    # params
    # right

    # calculate all bridged_shapes for case 1, then keep the params
    # to calculate case 2
    all_left = povcalnet.partition_by(['country', 'year'], as_dict=True)
    all_right = billy_pop.partition_by(['country', 'year'], as_dict=True)

    params_cache = dict()
    bridge_cache = dict()

    missing_country = set()
    failed_connect = set()

    geo_range = povcalnet['country'].unique().to_numpy().tolist()
    time_range = range(1980, 2101)

    for geo in geo_range:
        for t in time_range:
            left_shape = all_left.get((geo, t))
            if left_shape is None:  # no left shape
                continue
            right_shape = all_right.get((geo, t))
            if right_shape is None or right_shape['population'].sum() < 20:
                missing_country.add((geo, t))
                continue
            # right_shape = reindex_shape(right_shape)

            # scale = scale_dict[t]
            try:
                params, bridge_shape = calculate_bridge(left_shape, right_shape)
                params_cache[(geo, t)] = params
                bridge_cache[(geo, t)] = bridge_shape.clone()
            except KeyboardInterrupt:
                raise
            except:
                print(geo, t)
                failed_connect.add((geo, t))
    print("please double check above country/year pairs which the bridging failed. "
          "Ignore this message if there are no country year pairs above")

    # plot some results
    # plot_shape(bridge_cache[('chn', 2050)], alpha=.5)
    # plot_shape(_f(povcalnet, country='chn', year=2050), alpha=.5)
    # plt.show()

    # now calculate case 2
    # first we need to get the gini as parameter
    # NOTE: Here I am  still using pandas. maybe should change to polars too
    # gini = pd.read_csv('../../../ddf--gapminder--fasttrack/ddf--datapoints--gini_2100--by--country--time.csv')
    gini = pd.read_csv('../build/source/gapminder/gini.csv')
    gini = gini.set_index(['geo', 'time'])['gini_2100']

    params_cache_df = pd.DataFrame.from_dict(params_cache).T
    params_cache_df['gini'] = gini.loc[params_cache_df.index]
    params_cache_df.columns = ['slope', 'bridge_left_y', 'mountain_top_y', 'gini']
    params_cache_df['ptc_top'] = np.log(params_cache_df['bridge_left_y']) / np.log(params_cache_df['mountain_top_y'])

    # plt.scatter(params_cache_df.gini, params_cache_df.slope)
    # plt.show()
    # plt.scatter(params_cache_df.gini, params_cache_df.ptc_top)
    # plt.show()

    # divide gini into 3 categories (high, low, middle) and calculate the average slope and average
    # starting point from the data.
    low_gini_params = params_cache_df[params_cache_df['gini'] < 33]
    mid_gini_params = params_cache_df[params_cache_df['gini'].between(33, 50)] # note: inclusive = both
    high_gini_params = params_cache_df[params_cache_df['gini'] > 50]

    low_params = low_gini_params.agg({'slope': 'mean', 'ptc_top': 'mean'})
    mid_params = mid_gini_params.agg({'slope': 'mean', 'ptc_top': 'mean'})
    high_params = high_gini_params.agg({'slope': 'mean', 'ptc_top': 'mean'})

    # testing
    # df = all_left[('dom', 2018)]
    # df
    # bridge_shape = calculate_bridge_2(df, mid_params)
    # bridge_shape
    # _, ax = plt.subplots(1, 1)
    # plot(df[200:], bridge_shape[200:], log=True)
    # plt.show()

    # next run all missing countries
    no_bridge = set()
    for mc in missing_country:
        try:
            _gini = gini.loc[mc]
        except KeyError:
            continue
        if _gini < 33:
            _params = low_params.copy()
        elif _gini >= 33 and _gini <= 50:
            _params = mid_params.copy()
        else:
            _params = high_params.copy()

        # use higher slope for those missing right shapes
        _params['slope'] = _params['slope'] * 3

        try:
            left_shape = all_left[mc]
            bridge_shape = calculate_bridge_2(left_shape, _params)
        except:
            print(mc, _gini)
            raise

        if bridge_shape is not None:
            bridge_cache[mc] = bridge_shape
        else:
            no_bridge.add(mc)

    # check the bridge_cache
    for k, df in bridge_cache.items():
        pdf = all_left[k]
        assert pdf['bracket'].min() == df['bracket'].min()

    # bridge = bridge_cache[('swe', 2030)]
    # right = all_right[('swe', 2030)]
    # right
    # plot(bridge, right, log=True)

    # NOTE: len(bridge_cache) is not equal to len(all_left), because
    # there are shapes for 1800-1980 which are not involved in bridge process
    # len(bridge_cache)
    # len(all_left)
    all_bridges = pl.concat([
        df.with_columns(
            pl.lit(k[0]).alias('country'),
            pl.lit(k[1]).alias('year').cast(pl.Int32)
        )
        for k, df in bridge_cache.items()
    ]).select(
        ['country', 'year', 'bracket', 'population']
    ).sort(['country', 'year', 'bracket'])

    all_shapes = povcalnet.join(
        all_bridges,
        on=['country', 'year', 'bracket'],
        how='full',
        coalesce=True
    ).select(
        pl.col(['country', 'year', 'bracket']),
        pl.coalesce(pl.col(["population_right", "population"]), pl.lit(0)).alias('population'))

    all_shapes = all_shapes.sort(['country', 'year', 'bracket'])

    all_shapes.write_parquet('./bridged_shapes.parquet')

    print('making some plots for checking...')
    make_checking_plots(povcalnet, billy_pop, all_shapes)
    print('Done!')
