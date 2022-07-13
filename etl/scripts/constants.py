import numpy as np
import pandas as pd

all_brackets = np.logspace(-7, 13, 201, endpoint=True, base=2)
brackets_delta = 0.1  # it's (13 - (-7)) / 200

coverage_type_dtype = pd.CategoricalDtype(list('NAUR'), ordered=True)


# the linear version of weight function
# TODO: maybe add expontional version
def get_weights(y):
    """returns W(povcal), W(average shape)"""
    if y <= 1870:
        return (0, 1)
    if y > 1870 and y <= 1960:  # 1871 - 1960
        step = 0.01
        w = 0.1 + (1961 - y) * step
        return (1-w, w)
    if y > 1960 and y <= 1980:  # 1961 - 1980
        step = 0.005
        w = (1981 - y) * step
        return (1-w, w)
    if y > 1980 and y <= 2019:  # with in povcal period.
        return (0.995, 0.005)
    if y > 2019:
        if y > 2050:
            raise ValueError("year > 2050 not supportted")
        step = 0.005
        w = 0.005 + (y - 2019) * step
        return (1-w, w)


all_years = set(range(1800, 2051))
all_weights = dict([(x, get_weights(x)) for x in all_years])
