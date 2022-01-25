import pandas as pd
import numpy as np
import constants

usecols = [
    'CountryCode', 'CountryName', 'CoverageType', 'RequestYear',
    'HeadCount', 'ReqYearPopulation', 'Mean'
]


def load_file_preprocess(filename):
    coverage_type_dtype = constants.coverage_type_dtype
    df = pd.read_csv(filename, usecols=usecols,
                     dtype={'CoverageType': coverage_type_dtype})
    df = df.rename(
        columns={
            'CountryCode': 'country',
            'CoverageType': 'coverage_type',
            'RequestYear': 'year'
        })
    df = df.set_index(['country', 'year', 'coverage_type'])
    return df


def bracket_number_from_income(s, mountly=False, integer=True):
    # FIXME: double check if it should +1 to the result
    # because int(x) will drop the decimal part.
    if mountly:  # calculate daily income
        res = ((np.log2(s / 30) + 7) / constants.brackets_delta)
    else:
        res = ((np.log2(s) + 7) / constants.brackets_delta)
    if integer:
        return res.astype(int)
    return res
