import os
import pandas as pd
from functools import partial
from ddf_utils.str import format_float_digits
from smoothlib import run_smooth
from multiprocessing import Pool


source_dir = '../source'
bracket_file = os.path.join(source_dir, 'brackets.csv')

POOLSIZE = 3
formattor = partial(format_float_digits, digits=6)


def load_file_preprocess(filename, income_bracket):
    usecols = [
        'CountryCode', 'CountryName', 'CoverageType', 'RequestYear',
        'PovertyLine', 'HeadCount', 'ReqYearPopulation'
    ]
    df = pd.read_csv(os.path.join(source_dir, filename), usecols=usecols)
    df = df.rename(
        columns={
            'CountryCode': 'geo',
            'PovertyLine': 'income_bracket',
            'CoverageType': 'coverage_type',
            'RequestYear': 'year'
        })
    df['income_bracket'] = income_bracket
    df = df.set_index(['geo', 'year', 'coverage_type', 'income_bracket'])
    return df


def serve_datapoint(df, col):
    df = df.reset_index()
    df.columns = ['geo', 'year', 'coverage_type', 'income_bracket', col]
    df['geo'] = df['geo'].str.lower()
    df['coverage_type'] = df['coverage_type'].str.lower()
    df[col] = df[col].map(formattor)
    df.to_csv(
        f'../../ddf--datapoints--{col}--by--geo--year--coverage_type--income_bracket.csv',
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


def process(idx, df):
    """function for processing smoothing for each series"""
    print(idx)
    names = ['geo', 'year', 'coverage_type', 'income_bracket']
    idx_new = pd.MultiIndex.from_product([(idx[0],), (idx[1],), (idx[2],), range(100)], names=names)
    ser = df[idx]
    if ser.hasnans:
        if ser.dropna().empty:
            print("empty series")
            return pd.Series([])
        else:
            print('NaNs detected, fill them with zeros')
            ser = ser.fillna(0)
    res = func(ser)
    res.index = idx_new
    return res


def main():
    all_source_files = os.listdir(source_dir)
    all_source_files = [x for x in all_source_files if x.startswith('0')]
    all_source_files.sort()

    brackets = pd.read_csv(bracket_file)
    # use brackets end to indentity income_brackets
    brackets_entity_mapping = brackets['bracket_end'].to_dict()
    brackets_entity_mapping_reverse = dict([
        (v, k) for k, v in brackets_entity_mapping.items()
    ])

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

        # collect datapoints
        population_percentage = df_end['HeadCount'] - df_start['HeadCount']

        total_population_list.append(df_start['ReqYearPopulation'])
        population_percentage_list.append(population_percentage)

    # concat
    df_total_population = pd.concat(total_population_list).sort_index(
        level=['geo', 'year', 'coverage_type', 'income_bracket'])
    df_population_percentage = pd.concat(
        population_percentage_list).sort_index(
            level=['geo', 'year', 'coverage_type', 'income_bracket'])

    # FIX negitave values
    df_population_percentage[df_population_percentage < 0] = 0

    # adding a smooth indicator for population percentage
    df_pop_pct_smooth = df_population_percentage.copy()
    groups = df_pop_pct_smooth.groupby(level=['geo', 'year', 'coverage_type'])
    process_func = partial(process, df=df_pop_pct_smooth)
    with Pool(POOLSIZE) as p:
        result = p.map(process_func, groups.groups.keys())

    df_pop_pct_smooth = pd.concat(result)
    df_pop_pct_smooth = df_pop_pct_smooth.dropna(how='any')

    # computing population and population smoothed
    df_population = df_total_population * df_population_percentage * 1000000
    df_population_smooth = df_total_population * df_pop_pct_smooth * 1000000

    # serving datapoints
    serve_datapoint(df_population, 'population')
    serve_datapoint(df_population_smooth, 'population_smooth')
    serve_datapoint(df_population_percentage, 'population_percentage')
    serve_datapoint(df_pop_pct_smooth, 'population_percentage_smooth')


if __name__ == '__main__':
    main()
