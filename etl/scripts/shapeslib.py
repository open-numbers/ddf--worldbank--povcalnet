import os
import os.path as osp
import pandas as pd
import numpy as np

import constants
import etllib

# income_gini_df = pd.read_csv(
#     '../wip/income_gini_for_known_shape_countries.csv').set_index(['country', 'year']).sort_index()
# knows_shapes = pd.read_csv(
#     '../wip/smoothshape/ddf--datapoints--population_percentage--by--country--year--coverage_type--bracket.csv',
# )


def get_distances(income, gini, known_income_gini):
    """ calculate distances from one point to all known income/gini points as in 2D plane.
    the distance is calculated by sqrt(gini_distance^2 + income_distance^2)
    and gini_distance is linear scale distance and income_distance is log scale distance (base 2)

    NOTE: pre-process should be done to ensure log scale and column names in `known_shapes` DataFrame
    """
    gini_distances = np.abs(known_income_gini['gini'] - gini)
    income_distances = np.abs(known_income_gini['income'] - np.log2(income))

    res = np.sqrt(np.power(income_distances, 2) + np.power(gini_distances, 2))
    return res.sort_values(ascending=True).dropna()


def get_neighbors(income, gini, known_income_gini, n=50):
    """get nearest N neighbors. The distance is calcualted by `get_distance`

    returns: (number of countries in neighbors, neighbors list)
    """
    neis = list()
    ecs = set()
    res = get_distances(income, gini, known_income_gini).iloc[:n]
    for i, v in res.iteritems():
        ecs.add(i[0])
        neis.append(i)

    return (len(ecs), neis)


def get_shape(idx, known_shapes):
    """get shape from known shapes
    """
    df = known_shapes.loc[idx]
    for t in 'naur':
        if t in df['reporting_level'].values:
            # if t in 'ur':
            #     print(f'{idx}: using urban/rural data')
            df_nc = df[df['reporting_level'] == t]
            df_nc = df_nc.set_index('bracket')['population_percentage']
            return df_nc


def merge_nshapes(s_list):
    """function for merging shapes
    s_list: a list of shapes objects (pd.Series)
    """
    res = pd.concat(s_list, axis=1).fillna(0)
    return res.mean(axis=1)


def merge_nshapes_with_weights(s_list, weights):
    if np.sum(weights) != 1:
        raise ValueError('weights should sum up to 1')
    new_list = []
    for s, w in zip(s_list, weights):
        new_list.append(s * w)
    res = pd.concat(new_list, axis=1).fillna(0)
    return res.sum(axis=1)


def get_income_gini(idx, income_gini):
    i = income_gini.loc[idx, 'income']
    g = income_gini.loc[idx, 'gini']
    return i, g


def get_average_shape(c, y, shapes, neighbours):
    y = str(y)
    nei = neighbours[c][y]['neighbours']
    nei = [tuple(x) for x in nei]
    return shapes.loc[shapes.index.get_level_values(0).isin(nei)].groupby('bracket').sum() / 50


def get_average_shape2(c, y, shapes, all_country_year, neighbours):
    """this one is optimized for running speed, by adding pre-computed all available country_year series"""
    y = str(y)
    nei = neighbours[c][y]['neighbours']
    nei = [tuple(x) for x in nei]
    mask = all_country_year.isin(nei)
    return shapes.loc[mask].groupby('bracket').sum() / 50


def get_nearest_known_shape(country, year, known_shapes):
    try:
        df = known_shapes.loc[country]
    except KeyError:
        return None

    if year > 2017:
        nearest = df.index.get_level_values(0)[-1]
        # print(nearest)
    else:
        nearest = df.index.get_level_values(0)[0]
    return df.loc[nearest]


def shape_to_mountain(shape, income):
    bracket = etllib.bracket_number_from_income(income, integer=False)
    shape.index = shape.index + bracket
    # if 0 in shape.index.values:
    #     res = shape.loc[0:199]
    # else:
    #     res = shape.loc[:199]
    # if len(res) != 200:
        # print(f'not enough points in mixed shape: {idx}')
        # new_idx = pd.Index(range(200))
        # res = shape.reindex(new_idx, fill_value=0)
    res = shape.copy()
    return res


def get_estimated_mountain(idx, income, known_shapes, all_country_year, neighbours, n=50):
    country, year = idx
    wpov, was = constants.all_weights[year]

    first_known_shape = get_nearest_known_shape(country, year, known_shapes)
    if first_known_shape is None:
        return None

    if wpov == 1:
        mixed_shape = first_known_shape
    elif wpov == 0:
        mixed_shape = get_average_shape2(country, year, known_shapes, all_country_year, neighbours)
    else:
        average_shape = get_average_shape2(country, year, known_shapes, all_country_year, neighbours)
        mixed_shape = merge_nshapes_with_weights([first_known_shape, average_shape], (wpov, was))

    return shape_to_mountain(mixed_shape, income)
