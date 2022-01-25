# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from functools import partial
from ddf_utils.str import format_float_digits


# important steps
# b2 = b / b.shift(1)
# b2 = b2.shift(-1).fillna(1)[::-1]
# np.divide.accumulate(b2) * beg * 12
INCOME = pd.read_csv('../../ddf--datapoints--mean_income--by--country--year.csv')
GDP = pd.read_csv(
    '../../../ddf--gapminder--gdp_per_capita_cppp/ddf--datapoints--income_per_person_gdppercapita_ppp_inflation_adjusted--by--geo--time.csv')
FORMATTOR = partial(format_float_digits, digits=6)


def extrapolate(mean, gdp, how='back'):
    if how == 'back':
        begin = mean.iloc[0]
        begin_index = mean.index[0]
        growth = gdp.loc[:begin_index]
        growth = growth / growth.shift(1)
        growth = growth.shift(-1).fillna(1)[::-1]
        res = np.divide.accumulate(growth) * begin
        return res.loc[::-1]
    elif how == 'fore':
        last = mean.iloc[-1]
        last_index = mean.index[-1]
        growth = gdp.loc[last_index:]
        growth = growth.shift(-1) / growth
        growth = growth.shift(1).fillna(1)
        res = np.multiply.accumulate(growth) * last
        return res


def process(mean, gdp, key):
    m = mean.loc[key]
    g = gdp.loc[key]
    back = extrapolate(m, g, 'back')
    fore = extrapolate(m, g, 'fore')
    back.index.name = 'year'
    back.name = 'mean_income_extrapolated'
    fore.index.name = 'year'
    fore.name = 'mean_income_extrapolated'
    m.name = 'mean_income_extrapolated'
    res = pd.concat([back, m, fore])
    res = res[~res.index.duplicated(keep='first')]
    res.index = pd.MultiIndex.from_product([[key], res.index], names=['country', 'year'])
    return res


def main():
    countries = INCOME['country'].unique()
    geos = GDP['geo'].unique()
    mean_income = INCOME.set_index(['country', 'year'])['mean_income'] * 12
    gdp = GDP.set_index(['geo', 'time'])['income_per_person_gdppercapita_ppp_inflation_adjusted']

    res = list()
    for k in countries:
        if k in geos:
            res.append(process(mean_income, gdp, k))
        else:
            print(f'missing geo in gdp history: {k}')

    return pd.concat(res)


if __name__ == '__main__':
    res = main()
    res = res.map(FORMATTOR)
    res.to_csv('../../ddf--datapoints--mean_income_extrapolated--by--country--year.csv')
