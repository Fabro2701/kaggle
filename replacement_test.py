import numpy as np
import pandas as pd
import sys
import random


def test_df(original_df: pd.DataFrame, replace_null_values, columns):
    df_copy = original_df.copy(deep=True)
    results = {}
    indices = {}
    for column in columns:
        idx = df_copy[~df_copy[column].isna()].index
        nan_idx = random.sample(list(idx), int(len(idx) / 10))
        df_copy.loc[nan_idx, column] = np.nan
        indices[column] = nan_idx

    df_copy = replace_null_values(df_copy)

    for column in columns:
        idx = indices[column]
        if pd.api.types.is_numeric_dtype(df_copy[column].dtype):
            aux = abs(original_df.loc[idx, column] - df_copy.loc[idx, column])
            results[column] = aux.mean() / abs(original_df.loc[idx, column]).mean()
        else:
            results[column] = ((original_df.loc[idx, column] == df_copy.loc[idx, column]).sum()) / len(idx)
    return results

