# -*- coding: utf-8 -*-

"""Download various Gapminder indicators for computation
"""

import sys
import os
import gspread
import gspread_pandas
from gspread_pandas import Spread

# TODO: check file correctness after download

URLS = {
    'mean_income': {
        'docid': '1FQTKYJY55ZvHdcBrUbH5wlBA__eVC6p33pW0Ddqrl6Y',
        'sheet': 'data-for-countries-etc-by-year'
    },
    'gdp': {
        'docid': '1QRBztI6iSaGt9VUG9FuVVRCkeNsAxiqOGJ3Sr_gwh54',
        'sheet': 'data-for-countries-etc-by-year'
    },
    'gini': {
        'docid': '18oDZZV2T0DZYsx-5qa9yAsgfm1MhRxq6swhvLMp-iGw',
        'sheet': 'data-for-countries-etc-by-year'
    },
    'population': {
        'docid': '1c1luQNdpH90tNbMIeU7jD__59wQ0bdIGRFpbMm8ZBTk',
        'sheet': 'data-for-countries-etc-by-year'
    },
    'on_income_groups': {
        'docid': '1F4V12IJUAXFq3lgrtN92IWQrxmV4DnXlQAJhBItkyII',
        'sheet': 'data-for-countries-etc-by-year'
    },
    'wb_income_groups': {
        'docid': '1X6NenlJpv4aP2ApFGVmtiUdMCnqOcimpVelW5KcNgg4',
        'sheet': 'data-for-countries-etc-by-year'
    },
    'national_poverty_lines': {
        'docid': '1ip5eftTzJ3grJ-03fxqTyzr4nmDB9bPWjrIizBoy7Kg',
        'sheet': 'data-for-countries-etc-by-year'
    }
}

output_dir = 'source/gapminder/'


def download_gspread(k, info):
    spread = Spread(info['docid'])
    df = spread.sheet_to_df(sheet=info['sheet'], index=None)

    fname = os.path.join(output_dir, f'{k}.csv')
    df.to_csv(fname, index=False)


def download_all():
    for f, info in URLS.items():
        print(f'downloading {f} indicator...')
        download_gspread(f, info)


if __name__ == '__main__':
    # make dir
    os.makedirs(output_dir, exist_ok=True)

    if len(sys.argv) > 1:
        key = sys.argv[1]
        info = URLS[key]
        print(f'downloading {key} indicator...')
        download_gspread(key, info)
    else:
        download_all()
