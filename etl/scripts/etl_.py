"""
Notes:
- Poverty Line in povcal is in $/day
- mean income in fasttrack is also $/day
- brackets are defined with $/day
- But mean income in povcal is $/month. (But we are not using mean income from povcal)
"""

import os
import shutil
import pandas as pd
from functools import partial
from ddf_utils.str import format_float_digits
from smoothlib import run_smooth
import geogroup
from multiprocessing import Pool


source_dir = '../source/'
bracket_file = os.path.join(source_dir, 'fixtures', 'brackets.csv')

POOLSIZE = 3
formattor = partial(format_float_digits, digits=6)
coverage_type_dtype = pd.CategoricalDtype(list('NAUR'), ordered=True)
fixtures = [
    "ddf--entities--coverage_type.csv",
    "ddf--entities--geo--on_income_level.csv",
    "ddf--entities--geo--wb_income_group.csv",
    "ddf--entities--geo--west_and_rest.csv",
    "ddf--entities--geo--world_4regions.csv"
]


def load_file_preprocess(filename, income_bracket):
    usecols = [
        'CountryCode', 'CountryName', 'CoverageType', 'RequestYear',
        'PovertyLine', 'HeadCount', 'ReqYearPopulation'
    ]
    df = pd.read_csv(os.path.join(source_dir, filename), usecols=usecols,
                     dtype={'CoverageType': coverage_type_dtype})
    df = df.rename(
        columns={
            'CountryCode': 'country',
            'PovertyLine': 'income_bracket',
            'CoverageType': 'coverage_type',
            'RequestYear': 'year'
        })
    df['income_bracket'] = income_bracket
    df = df.set_index(['country', 'year', 'coverage_type', 'income_bracket'])
    return df


def serve_datapoint(df, index_cols, data_col, is_float=True):
    df = df.reset_index()
    columns = index_cols.copy()
    columns.extend([data_col])
    df.columns = columns
    df['country'] = df['country'].str.lower()
    if "coverage_type" in df.columns:
        df['coverage_type'] = df['coverage_type'].str.lower()
    if is_float == 'float':   # population percentage datapoint
        df[data_col] = df[data_col].map(formattor)
    else:  # population count datapoint
        df[data_col] = df[data_col].astype(int)
    by = '--'.join(index_cols)
    df.to_csv(
        f'../../ddf--datapoints--{data_col}--by--{by}.csv',
        index=False)
    return


def func(x):
    """function to smooth a series"""
    x = x.reset_index(drop=True)
    # run smoothing
    std = x.std()
    res = run_smooth(x, 10, 1)
    if std < 0.021:
        res = run_smooth(res, 8, 1)
        res = run_smooth(res, 8, 1)
        res = run_smooth(res, 5, 0)
        res = run_smooth(res, 5, 0)
    else:
        res = run_smooth(res, 8, 0)
        res = run_smooth(res, 8, 0)
        res = run_smooth(res, 5, 0)
        res = run_smooth(res, 5, 0)
    # also, make sure it will sum up to 100%
    res = res / res.sum()
    return res


def process(idx, ser):
    """function to process smoothing for each series"""
    # print(idx)
    names = ['country', 'year', 'coverage_type', 'income_bracket']
    subser = ser[idx]
    res = func(subser)
    # ensure that the output has 100 records.
    idx_new = pd.MultiIndex.from_product([(idx[0],), (idx[1],), (idx[2],), range(100)], names=names)
    res.index = idx_new  # raise error if length doesn't match
    return res


def main():
    all_source_files = os.listdir(source_dir)
    all_source_files = [x for x in all_source_files if x.startswith('0')]
    all_source_files.sort()

    source_pairs = list(
        zip(all_source_files[:-1], all_source_files[1:]
            ))  # a list of file name pairs (bracket start, bracket end)

    total_population_list = []
    population_percentage_list = []

    country_list = []

    for i, files in enumerate(source_pairs):
        start_file, end_file = files
        df_start = load_file_preprocess(start_file, i)
        df_end = load_file_preprocess(end_file, i)

        # collect geos
        geo1 = df_start['CountryName'].reset_index(
            [1, 2, 3], drop=True).drop_duplicates()
        geo2 = df_end['CountryName'].reset_index([1, 2, 3],
                                                 drop=True).drop_duplicates()
        country_list.append(geo1)
        country_list.append(geo2)

        # FIX: drop all rows where Headcount is NaN
        df_end = df_end.dropna(subset=['HeadCount'])
        df_start = df_start.dropna(subset=['HeadCount'])
        # collect datapoints
        population_percentage = df_end['HeadCount'] - df_start['HeadCount']
        population_percentage_list.append(population_percentage)
        total_population_list.append(df_start['ReqYearPopulation'])

    # concat
    df_total_population = pd.concat(total_population_list).sort_index(
        level=['country', 'year', 'coverage_type', 'income_bracket'])
    df_population_percentage = pd.concat(
        population_percentage_list).sort_index(
            level=['country', 'year', 'coverage_type', 'income_bracket'])
    # FIX: negitave values
    df_population_percentage[df_population_percentage < 0] = 0

    # map geo to open-numbers domain because we are going to calculate
    # geo groups aggragates defined in open numbers namespace
    countries = pd.concat(country_list).drop_duplicates()
    countries = countries.reset_index()
    countries['on_geo'] = countries['CountryName'].map(geogroup.syn_mapping)
    countries_mapping = countries.set_index('country')['on_geo'].to_dict()
    new_idx = df_total_population.index.levels[0].map(countries_mapping)
    df_total_population.index = df_total_population.index.set_levels(
        new_idx, level='country')
    new_idx = df_population_percentage.index.levels[0].map(countries_mapping)
    df_population_percentage.index = df_population_percentage.index.set_levels(
        new_idx, level='country')

    # adding a smooth indicator for population percentage
    df_pop_pct_smooth = df_population_percentage.copy()
    groups = df_pop_pct_smooth.groupby(level=['country', 'year', 'coverage_type'])

    process_func = partial(process, ser=df_pop_pct_smooth)
    with Pool(POOLSIZE) as p:
        result = p.map(process_func, groups.groups.keys())
    df_pop_pct_smooth = pd.concat(result)
    df_pop_pct_smooth = df_pop_pct_smooth.dropna(how='any')

    # computing population and population smoothed
    # it seems there is an interesting bug on pandas 1.2.4:  if I do
    # multiplication as following commented out lines, CHN 2018/2019 data will
    # be missing in df_population_smooth
    # df_population = df_total_population * df_population_percentage * 1000000
    # df_population_smooth = df_total_population * df_pop_pct_smooth * 1000000
    # So I have to concat them first
    df_combined = pd.concat([df_total_population,
                             df_population_percentage,
                             df_pop_pct_smooth], axis=1)
    df_combined.columns = ['total_population', 'population_percentage', 'pop_pct_smooth']
    df_population = df_combined.total_population * df_combined.population_percentage * 1000000
    df_population_smooth = df_combined.total_population * df_combined.pop_pct_smooth * 1000000

    # create datapoints without the coverage_type dimension
    # because datapoints are sorted, and coverage_type is a
    # category where N>A>U>R, so National data will be selected
    # if it's available, then Aggregated and so on
    df_population_nocov = df_population.groupby(
        level=['country', 'year', 'income_bracket']).first()
    df_population_percentage_nocov = df_population_percentage.groupby(
        level=['country', 'year', 'income_bracket']).first()
    df_population_smooth_nocov = df_population_smooth.groupby(
        level=['country', 'year', 'income_bracket']).first()
    df_pop_pct_smooth_nocov = df_pop_pct_smooth.groupby(
        level=['country', 'year', 'income_bracket']).first()

    # TODO: create datapoints by groups
    # we only do this for smoothed and w/o coverage_type dimension indicators
    # I have tried to do smoothing after grouping, the result is not much different
    # countries
    # geogroup.dosomething(df_population_smooth_nocov)

    # serving datapoints
    idx_cols = ['country', 'year', 'coverage_type', 'income_bracket']
    serve_datapoint(df_population, idx_cols, 'population', is_float=False)
    serve_datapoint(df_population_smooth, idx_cols, 'population_smooth', is_float=False)
    serve_datapoint(df_population_percentage, idx_cols, 'population_percentage', is_float=True)
    serve_datapoint(df_pop_pct_smooth, idx_cols, 'population_percentage_smooth', is_float=True)

    idx_cols = ['country', 'year', 'income_bracket']
    serve_datapoint(df_population_nocov, idx_cols, 'population', is_float=False)
    serve_datapoint(df_population_smooth_nocov, idx_cols, 'population_smooth', is_float=False)
    serve_datapoint(df_population_percentage_nocov, idx_cols, 'population_percentage', is_float=True)
    serve_datapoint(df_pop_pct_smooth_nocov, idx_cols, 'population_percentage_smooth', is_float=True)

    # serving domains
    # income bracket
    brackets = pd.read_csv(bracket_file, dtype=str)
    bracket_domain = brackets.reset_index()
    bracket_domain['name'] = brackets['bracket_start'] + ' - ' + brackets['bracket_end']
    bracket_domain = bracket_domain[['index', 'name']]
    bracket_domain.columns = ['income_bracket', 'name']
    bracket_domain.to_csv('../../ddf--entities--income_bracket.csv', index=False)

    # coverage type, geo groups are fixtures
    for f in fixtures:
        shutil.copyfile(
            os.path.join('../source', 'fixtures', f),
            os.path.join('../../', f)
        )

    # country

    # serving concepts


if __name__ == '__main__':
    main()
