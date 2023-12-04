"""Just copy national poverty lines dataset into this dataset.
"""

import os.path as osp
import polars as pl


data = pl.read_csv('../build/source/gapminder/national_poverty_lines.csv')

data.select(
    pl.col('geo').alias('country'),
    pl.col('time'),
    pl.col('National poverty line').alias('poverty_line')
).write_csv('ddf/poverty_rates/ddf--datapoints--poverty_line--by--country--time.csv')

print('done.')
