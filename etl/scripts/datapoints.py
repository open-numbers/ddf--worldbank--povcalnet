"""functions for creating datapoints for this dataset

TBD:
- mean income by country/year
- median income by country/year
- income_mountain by country/year/bracket
"""

import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from ddf_utils.str import format_float_digits
from functools import partial

import etllib
import smoothlib
import shapeslib
run_smooth = smoothlib.run_smooth
formattor = partial(format_float_digits, digits=6)

income_file = '../../../ddf--gapminder--fasttrack/ddf--datapoints--mincpcap_cppp--by--country--time.csv'
gini_file = '../../../ddf--gapminder--fasttrack/ddf--datapoints--gini--by--geo--time.csv'


# step1: read all source files
def step1():
    res = dict()
    for f in os.listdir('../source/'):
        if f.endswith('.csv'):
            fn = f.split('.')[0]
            bracket = fn.lstrip('0')
            if bracket == '':
                bracket = 0
            else:
                bracket = int(bracket)
            res[bracket] = etllib.load_file_preprocess(os.path.join('../source/', f))
    return res


# step 2: remove nans
def step2(res1):
    res = dict()
    nans = set()
    for k, df in res1.items():
        if df['HeadCount'].hasnans:
            idxs = df[pd.isnull(df['HeadCount'])].index.unique()
            nans = nans.union(set(idxs))
        res[k] = df.dropna(how='any', subset=['HeadCount'])
    if len(nans) > 0:
        print("WARNING: NaNs detected in these datapoints, dropping them")
        for i in nans:
            print(i)
    return res


# step3: subtract and get bracket head count, and concat them to DataFrame
def step3(res2):
    res3 = list()
    for i in range(1, 201):
        df1 = res2[i]
        df2 = res2[i-1]
        df3 = df1[['HeadCount']] - df2[['HeadCount']]
        df3['bracket'] = i - 1
        df3 = df3.set_index('bracket', append=True)
        res3.append(df3)
    return pd.concat(res3)


# step4: fix negative values
def step4(res3):
    res4 = list()
    gs = res3.groupby(['country', 'year', 'coverage_type'])
    for g in gs.groups.keys():
        df = gs.get_group(g)
        s = df['HeadCount'].copy()
        todrop = set()
        if np.any(s < 0):  # if negative values exists
            where = np.where(s < 0)[0]
            for w in where:
                if w != 199:
                    todrop.add(w+1)
                if w != 0:
                    todrop.add(w-1)
                todrop.add(w)
            s.iloc[list(todrop)] = np.nan
            res4.append(s)
        else:
            res4.append(s)
    return pd.concat(res4)

# step5: get smoothed shape, ensure the shape sum up to 100%
def func(x):
    """function to smooth a series"""
    if x.hasnans:
        x = x.interpolate()
        if pd.isnull(x.iloc[0]):
            x = x.fillna(0)
    # x = x.reset_index(drop=True)
    # run smoothing
    std = x.std()
    if std < 0.010:
        res = run_smooth(x, 20, 6)
        res = run_smooth(res, 16, 2)
        res = run_smooth(res, 16, 1)
        res = run_smooth(res, 10, 1)
        res = run_smooth(res, 10, 0)
    elif std < 0.012:
        res = run_smooth(x, 20, 3)
        res = run_smooth(res, 16, 2)
        res = run_smooth(res, 16, 1)
        res = run_smooth(res, 10, 0)
        res = run_smooth(res, 8, 0)
    else:
        res = run_smooth(x, 20, 1)
        res = run_smooth(res, 16, 1)
        res = run_smooth(res, 16, 0)
        res = run_smooth(res, 10, 0)
        res = run_smooth(res, 8, 0)
    # also, make sure it will sum up to 100%
    if res.min() < 0:
        res = res - res.min()
    res = res / res.sum()
    return res


def process(ser):
    idx = ser.index
    try:
        s_new = func(ser)
        s_new.index = idx
    except:
        print(idx[0])
        return pd.Series()
    return s_new


def step5(res4):
    gs = res4.groupby(['country', 'year', 'coverage_type'])
    to_smooth = list()
    for g, df in gs:
        to_smooth.append(df)
    print(len(to_smooth))
    with Pool(11) as p:
        res5 = p.map(process, to_smooth)
    return pd.concat(res5)


# step6: renaming and make it DDF valid
# also change xkx to kos for country, to align with open-numbers
def step6(res5):
    res6 = res5.copy()
    res6.name = 'population_percentage'
    res6 = res6.reset_index()
    res6['country'] = res6['country'].map(str.lower)
    res6['country'] = res6['country'].replace({'xkx': 'kos'})
    res6['coverage_type'] = res6['coverage_type'].map(str.lower)
    return res6.set_index(['country', 'year', 'coverage_type'])


# step7: insert bracket info, income data comes from fasttrack
def step7(df):
    income = pd.read_csv(income_file).set_index(['country', 'time'])
    income.index.names = ['country', 'year']
    income.columns = ['income']

    res = list()
    gs = df.groupby(['country', 'year', 'coverage_type'])
    for g, gdf in gs:
        df_ = gdf.copy()
        g_ = (g[0], g[1])
        try:
            m = income.loc[g_, 'income']
        except KeyError:
            print(f"missing: {g_}")
            continue
        b = etllib.bracket_number_from_income(m)
        df_['bracket'] = df_['bracket'] - b
        res.append(df_)

    return pd.concat(res)


# step8: get average shapes and estimated shapes
def step8(df):
    pass


# FIXME: don't use resN.

if __name__ == '__main__':
    res1 = step1()
    res2 = step2(res1)
    res3 = step3(res2)
    res4 = step4(res3)
    res4.to_csv('../wip/preprocessed.csv')
    res5 = step5(res4)
    res6 = step6(res5)
    res7 = step7(res6)
    # print(res5.head())
    # res7['population_percentage'] = res7['population_percentage'].map(formattor)
    res7.to_csv(
        '../wip/smoothshape/ddf--datapoints--population_percentage--by--country--year--coverage_type--bracket.csv')
