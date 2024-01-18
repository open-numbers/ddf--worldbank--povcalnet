# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import shapeslib
from multiprocessing import Pool
import psutil
import json

max_cpu = psutil.cpu_count(logical=False) - 1

income_file = 'source/gapminder/mean_income.csv'
gini_file = 'source/gapminder/gini.csv'
known_shapes_file = 'povcal_country_year.csv'


def get_distances_res(v):
    i = v[0]
    g = v[1]
    cno, neis = shapeslib.get_neighbors(i, g, known_income_gini)
    return (i, g), (cno, neis)


if __name__ == '__main__':

    income = pd.read_csv(income_file)
    gini = pd.read_csv(gini_file)
    known_shapes = pd.read_csv(known_shapes_file)

    income = income.set_index(['geo', 'time'])[['Average daily income per capita']]
    income.index.names = ['country', 'year']
    income.columns = ['income']

    gini = gini.set_index(['geo', 'time'])[['gini_2100']]
    gini.index.names = ['country', 'year']
    gini.columns = ['gini']

    income_gini = pd.concat([income, gini], axis=1)
    income_gini = income_gini.dropna(how='any')

    income_gini_noc = income_gini.reset_index(drop=True).drop_duplicates()

    # below countries are not available in income/gini data
    mask = known_shapes['country'].isin(['mhl', 'nru', 'twn', 'kos'])
    try:
        known_shapes_list = known_shapes[~mask].values.tolist()
        known_shapes_list = [tuple(x) for x in known_shapes_list]
        known_income_gini = income_gini.loc[known_shapes_list]
    except KeyError:
        print("could not find some countries in povcalnet data, please exclude them.")
        raise

    with Pool(max_cpu) as p:
        res_distances = p.map(get_distances_res, income_gini_noc.values)

    res_distances_dict = dict(res_distances)

    all_neighbours = []
    for cy in income_gini.index.values:
        i, g = shapeslib.get_income_gini(cy, income_gini)
        all_neighbours.append((cy, res_distances_dict[(i, g)]))

    all_neighbours = dict(all_neighbours)
    all_neighbours_json = dict()

    for k, v in all_neighbours.items():
        c, y = k
        if c not in all_neighbours_json.keys():
            all_neighbours_json[c] = dict()
        all_neighbours_json[c][y] = {
            "countries": v[0],
            "neighbours": [list(x) for x in v[1]]
        }

    k = json.dumps(all_neighbours_json)

    with open('neighbours_list.json', 'w') as f:
        f.write(k)
        f.close()
