import numpy as np
import pandas as pd


coverage_type_dtype = pd.CategoricalDtype(list('NAUR'), ordered=True)


# the slope and bias for interest rate calculation
a, b = (-0.022919607843137433, 47.1155565826334)

# the inflation rate 2011 = 1
rs = [1.548, 1.565, 1.553, 1.517, 1.464,
      1.445, 1.417, 1.369, 1.347, 1.346, 1.357,
      1.337, 1.289, 1.262, 1.252, 1.198, 1.126,
      1.088, 1.070, 1.065, 1.041, 1.032, 1.022,
      1.000, 0.974, 0.963, 0.948, 0.954, 0.955,
      0.937, 0.895, 0.876, 0.859, 0.82, 0.77]
rates = pd.Series(rs, index=range(1988, 2023))


def get_inflation_rate(y):
    if y in rates.index:
        return rates.loc[y]
    else:
        return a * y + b


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
        step = 0.005
        w = 0.005 + (y - 2019) * step
        return (1-w, w)


all_years = list(range(1800, 2101))
all_weights = dict([(x, get_weights(x)) for x in all_years])
