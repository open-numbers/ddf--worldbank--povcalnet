import os
import numpy as np
import pandas as pd
from functools import partial
from ddf_utils.str import format_float_digits
from scipy.ndimage import gaussian_filter1d
import scipy.signal as sg

source_dir = '../source'
bracket_file = os.path.join(source_dir, 'brackets.csv')

formattor = partial(format_float_digits, digits=6)


def load_file_preprocess(filename, income_bracket):
    usecols = ['CountryCode', 'CountryName', 'CoverageType', 'RequestYear', 'PovertyLine', 'HeadCount', 'ReqYearPopulation']
    df = pd.read_csv(os.path.join(source_dir, filename), usecols=usecols)
    df = df.rename(columns={'CountryCode': 'geo',
                                            'PovertyLine': 'income_bracket',
                                            'CoverageType': 'coverage_type',
                                            'RequestYear': 'year'})
    df['income_bracket'] = income_bracket
    df = df.set_index(['geo', 'year', 'coverage_type', 'income_bracket'])
    return df


def serve_datapoint(df, col):
    df = df.reset_index()
    df.columns = ['geo', 'year', 'coverage_type', 'income_bracket', col]
    df['geo'] = df['geo'].str.lower()
    df['coverage_type'] = df['coverage_type'].str.lower()
    # TODO: notify if there are negatives
    # reset all negatives to zeros
    df.loc[df[col] < 0, col] = 0
    df[col] = df[col].map(formattor)
    df.to_csv(f'../../ddf--datapoints--{col}--by--geo--year--coverage_type--income_bracket.csv', index=False)
    return


def main():
    all_source_files = os.listdir(source_dir)
    all_source_files = [x for x in all_source_files if x.startswith('0')]
    all_source_files.sort()

    brackets = pd.read_csv(bracket_file)
    # use brackets end to indentity income_brackets
    brackets_entity_mapping = brackets['bracket_end'].to_dict()
    brackets_entity_mapping_reverse = dict([(v, k) for k, v in brackets_entity_mapping.items()])
    
    source_pairs = list(zip(all_source_files[:-1], all_source_files[1:]))  # a list of file name pairs (bracket start, bracket end)
    
    total_population_list = []
    population_percentage_list = []

    country_list = []

    for i, files in enumerate(source_pairs):
        start_file, end_file = files
        df_start = load_file_preprocess(start_file, i)
        df_end = load_file_preprocess(end_file, i)
    
        # collect geos
        geo1 = df_start['CountryName'].reset_index([1, 2, 3], drop=True).drop_duplicates()
        geo2 = df_end['CountryName'].reset_index([1, 2, 3], drop=True).drop_duplicates()
        country_list.append(geo1)
        country_list.append(geo2)
    
        # collect datapoints
        population_percentage = df_end['HeadCount'] - df_start['HeadCount']  
        
        total_population_list.append(df_start['ReqYearPopulation'])
        population_percentage_list.append(population_percentage)

    # concat
    df_total_population = pd.concat(total_population_list).sort_index(level=['geo', 'year', 'coverage_type', 'income_bracket'])
    df_population_percentage = pd.concat(population_percentage_list).sort_index(level=['geo', 'year', 'coverage_type', 'income_bracket'])

    # adding a smooth indicator
    def func(x):
        # res = gaussian_filter1d(x, 2, truncate=1/4, order=2)
        res = sg.savgol_filter(x, 5, 1, mode='constant')
        return pd.Series(res, index=x.index)
        
    df_pop_pct_smooth = (df_population_percentage
                                        .groupby(level=['geo', 'year', 'coverage_type'])
                                        .apply(func))
    
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