# -*- coding: utf-8 -*-

import numpy as np
import polars as pl

import bracketlib


step = bracketlib.get_bracket_step(500)


# let's produce a 1050 version
all_brackets = np.arange(1050)

upper = bracketlib.income_from_bracket(all_brackets, step, 'upper', integer=False)
lower = bracketlib.income_from_bracket(all_brackets, step, 'lower', integer=False)


ent = pl.DataFrame(
    {
        'income_bracket_1050': all_brackets,
        'bracket_start': lower,
        'bracket_end': upper
    }
)


ent.write_csv('ddf/ddf--entities--income_bracket_1050.csv')


# resample to 105
ent2 = ent.with_columns(
    (pl.col('income_bracket_1050') / 10).cast(pl.Int32).alias('income_bracket_105')
).groupby(['income_bracket_105'], maintain_order=True).agg(
    pl.col('bracket_start').first(),
    pl.col('bracket_end').last()
).sort('income_bracket_105')


ent2.write_csv('ddf/ddf--entities--income_bracket_105.csv')
