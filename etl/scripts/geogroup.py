"""
"""

import os
import pandas as pd


wb_groups = pd.read_csv('../source/fixtures/wb_income_groups.csv')
on_groups = pd.read_csv('../source/fixtures/ddf--entities--geo--on_income_level.csv')
west_rest_groups = pd.read_csv('../source/fixtures/west_and_rest.csv')
synonyms = pd.read_csv('../source/fixtures/ddf--open_numbers/ddf--synonyms--geo.csv')
syn_mapping = synonyms.set_index('synonym')['geo'].to_dict()


def groups_over_time(df, col):
    df_copy = df.copy()
    df_copy['newindex'] = df['geo'] + '-' + df['year'].astype(str)
    return df_copy.set_index('newindex')[col].to_dict()


def translate_with_time_series(df: pd.DataFrame, mapping):
    df_copy = df.copy()
    df_copy['newindex'] = df['geo'] + '-' + df['year'].astype(str)
    df_copy['geo'] = df_copy['newindex'].map(mapping)

    return df_copy.drop(columns=['newindex'])


def translate_with_static_dict(df, mapping):
    df_copy = df.copy()
    df_copy['geo'] = df_copy['geo'].map(mapping)
    return df_copy


def create_group_datapoints(df, group, col='geo'):
    if group in ['wb_income_group', 'on_income_level']:
        mapping = groups_over_time(wb_groups, "WB's 4 income levels")

    elif group in ['west_and_rest', 'world_4regions']:
        pass
    else:
        raise NotImplementedError(f"not supported group: {group}")
