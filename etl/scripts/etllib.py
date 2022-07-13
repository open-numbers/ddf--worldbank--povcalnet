import pandas as pd
import numpy as np
import constants

usecols = [
    'country_code', 'country_name', 'reporting_level', 'reporting_year',
    'headcount', 'reporting_pop', 'mean'
]


def load_file_preprocess(filename):
    coverage_type_dtype = constants.coverage_type_dtype
    df = pd.read_csv(filename, usecols=usecols,
                     dtype={'CoverageType': coverage_type_dtype})
    df = df.rename(
        columns={
            'country_code': 'country',
            'reporting_year': 'year'
        })
    if np.any(df.duplicated(subset=['country', 'reporting_level', 'year'])):
        print(f"{filename} has duplicated entries")
        # df = df.drop_duplicates(subset=['country', 'reporting_level', 'year'])
    df = df.set_index(['country', 'year', 'reporting_level'])
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
