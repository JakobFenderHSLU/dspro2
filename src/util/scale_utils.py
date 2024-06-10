from typing import Tuple

import pandas as pd


def convert_to_small(train_df: pd.DataFrame, val_df: pd.DataFrame, label_count: int = 7) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert the DataFrames to small DataFrames for training and validation
    :param train_df: DataFrame containing training data
    :param val_df: DataFrame containing validation data
    :return: small DataFrames for training and validation
    """
    train_df = train_df.copy()
    val_df = val_df.copy()

    cols = train_df.columns[:label_count + 1]
    train_df = train_df[cols]
    val_df = val_df[cols]

    # removed all columns where species_name is everywhere 0
    condition_exists_train = train_df.iloc[:, 1:].sum(axis=1) > 0
    train_df = train_df[condition_exists_train]

    condition_exists_val = val_df.iloc[:, 1:].sum(axis=1) > 0
    val_df = val_df[condition_exists_val]

    # reset indexes
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    return train_df, val_df


def convert_to_debug(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert the DataFrames to debug DataFrames for training and validation
    :param train_df: DataFrame containing training data
    :param val_df: DataFrame containing validation data
    :return: debug DataFrames for training and validation
    """
    new_train_df = pd.DataFrame()
    new_val_df = pd.DataFrame()

    # get first 3 columns
    cols = train_df.columns[1:4]

    for col in cols:
        train_entry = train_df[train_df[col] == 1].iloc[0, :4]
        train_entry_df = pd.DataFrame(train_entry).T
        new_train_df = pd.concat([new_train_df, train_entry_df], axis=0)

        val_entry = val_df[val_df[col] == 1].iloc[0, :4]
        val_entry_df = pd.DataFrame(val_entry).T
        new_val_df = pd.concat([new_val_df, val_entry_df], axis=0)

    return new_train_df, new_val_df
