"""
Bracket related calculations

NOTE:
1. all income in these functions are daily income.
2. poverty line is exclusive. so poverty line $1 means <1$/day
"""

import math
import numpy as np
import pandas as pd


def get_bracket_step(totalsize):
    """return the bracket in log scale

    totalsize means how many brackets totally in the range [(2**(-7)), (2**13)]
    """
    bracket_step = (13 - (-7)) / totalsize
    return bracket_step


def income_from_bracket(b, bracket_step, bound='upper', integer=True):
    """return the lower bound or upper bound income of a bracket.

    NOTE:
      - b can be an array because we use np.power here
    """
    if bound == 'upper':
        exponent = (-7) + (b+1) * bracket_step
    elif bound == 'lower':
        exponent = (-7) + b * bracket_step
    else:
        raise ValueError("bound must be `upper` or `lower`")
    if integer:
        return np.power(2, exponent).astype(int)
    else:
        return np.power(2, exponent)


def bracket_from_income(inc, bracket_step):
    """given a income (poverty line), return the bracket number it belongs to

    which means this income level is in between the lower and upper bound
    of that bracket.
    """
    loc = math.log2(inc)
    res = (loc + 7) / bracket_step
    if res < 0:
        if integer:
            return 0
        return 0.0

    # if integer:  # we should always output integer
    res = int(math.floor((loc + 7) / bracket_step))

    return res
