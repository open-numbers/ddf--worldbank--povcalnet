# -*- coding: utf-8 -*-

import pandas as pd
from ddf_utils.str import format_float_digits
from functools import partial


SOURCE_FILE = '../source/0000.csv'
SYNONYM_FILE = '../source/fixtures/ddf--open_numbers/ddf--synonyms--geo.csv'
FORMATTOR = partial(format_float_digits, digits=6)
coverage_type_dtype = pd.CategoricalDtype(list('NAUR'), ordered=True)


def main():
    # 1. read source file. mean and median are not related to poverty line, so using one file is enough
    # 2. read the synonym  translate CountryName -> open-numbers Geo
    # 3. change column names, change coverage_type to lowercase
    # 4. create one indicator with the coverage_type variable
    # 5. create one indicator without the coverage_type variable.
    # TODO: median income contains -1, drop them
    cols = ['CountryName', 'RequestYear', 'CoverageType', 'Mean', 'Median']
    df = pd.read_csv(SOURCE_FILE, dtype={'CoverageType': coverage_type_dtype}, usecols=cols)
    synonym = pd.read_csv(SYNONYM_FILE)
    countrydict = synonym.set_index('synonym')['geo'].to_dict()
    df = df[cols]
    df.columns = ['country_name', 'year', 'coverage_type', 'mean_income', 'median_income']
    df['country'] = df['country_name'].map(countrydict)
    df = df.sort_values(by=['country', 'year', 'coverage_type'])
    df['coverage_type'] = df['coverage_type'].astype(str)
    df['coverage_type'] = df['coverage_type'].str.lower()

    mean = df[['country', 'year', 'coverage_type', 'mean_income']].copy()
    mean['mean_income'] = mean['mean_income'].map(FORMATTOR)
    print(mean.head())

    median = df[['country', 'year', 'coverage_type', 'median_income']].copy()
    median['median_income'] = median['median_income'].map(FORMATTOR)
    print(median.head())

    mean_nocov = mean.groupby(by=['country', 'year']).first()['mean_income']
    print(mean_nocov.head())
    median_nocov = median.groupby(by=['country', 'year']).first()['median_income']
    print(median_nocov.head())

    mean.dropna(how='any').to_csv('../../ddf--datapoints--mean_income--by--country--year--coverage_type.csv', index=False)
    median.dropna(how='any').to_csv('../../ddf--datapoints--median_income--by--country--year--coverage_type.csv', index=False)

    mean_nocov.dropna().to_csv('../../ddf--datapoints--mean_income--by--country--year.csv')
    median_nocov.dropna().to_csv('../../ddf--datapoints--median_income--by--country--year.csv')


if __name__ == '__main__':
    main()
    print('done.')
