import pandas as pd
from functools import partial
from ddf_utils.str import format_float_digits


formattor = partial(format_float_digits, digits=6)


def f(x):
    split = x.split(' - ')
    return ' - '.join([split[0], split[-1]])


def f2(x):
    name = x.name
    res = x + x.shift(-1)
    res = res.iloc[::2].reset_index(drop=True)
    res.name = name
    res.index.name = 'income_bracket_50'
    return res


def process(name, geo):
    filename = f'../../ddf--datapoints--{name}--by--{geo}--year--income_bracket.csv'
    df = pd.read_csv(filename, dtype=str)
    df[name] = df[name].astype(float)
    res = df.groupby([geo, 'year'])[name].apply(f2)
    res = res.astype(int)
    outfilename = filename.replace('income_bracket', 'income_bracket_50')
    res.to_csv(outfilename)
    return res


def process_2(name, bracket, geo):
    filename = f'../../ddf--datapoints--{name}--by--{geo}--year--{bracket}.csv'
    df = pd.read_csv(filename, dtype=str)
    res = df.groupby([geo, 'year'])[name].agg(lambda x: x.str.cat(sep=','))
    if bracket == 'income_bracket':
        name = 'income_mountain_100bracket_shape_for_log'
    else:
        name = 'income_mountain_50bracket_shape_for_log'
    res.name = name
    outfile = f'../../ddf--datapoints--{name}--by--{geo}--year.csv'
    res.to_csv(outfile)
    return res
