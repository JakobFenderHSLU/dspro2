import os
import argparse

import pandas as pd


def split_data(path: str = "./../../input/scrape/", split_ratio: [] = None, seed: int = 42):
    """
    Split the data into training, validation, and test sets based on the provided split ratio. Save the resulting
    DataFrames as CSV files in the specified directory.
    :param path: The path to the directory containing the data files.
    :param split_ratio: The ratio in which to split the data. Default is [0.6, 0.2, 0.2].
    :param seed: The seed value for the random number generator. Default is 42.
    :return: A DataFrame containing the split data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory {path} not found")

    folders = os.listdir(path)
    folders = [f for f in folders if os.path.isdir(os.path.join(path, f))]

    current_species_id = 0
    df_dict = {"species_id": [], "species_name": [], "file_path": []}

    for folder in folders:
        files = os.listdir(os.path.join(path, folder))
        files = [f for f in files if f.endswith('.mp3')]

        if len(files) == 0:
            print(f"Warning: No files found in {folder}")

        for file in files:
            file_path = str(os.path.join(path, folder, file))
            df_dict["species_id"].append(current_species_id)
            df_dict["species_name"].append(folder)
            df_dict["file_path"].append(file_path)

        current_species_id += 1

    df = pd.DataFrame(df_dict)

    if split_ratio is None:
        raise ValueError("Split ratio must be provided")

    if len(split_ratio) != 3:
        raise ValueError("Split ratio must contain 3 values")
    if sum(split_ratio) != 1:
        raise ValueError("Split ratio values must sum to 1")

    # ensure equal split for every species
    species_ids = df["species_id"].unique()
    train_ids = []
    val_ids = []
    test_ids = []
    for species_id in species_ids:
        species_df = df[df["species_id"] == species_id]
        species_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        species_df_index = species_df.index.to_list()
        n = len(species_df_index)
        train_size = int(split_ratio[0] * n)
        val_size = int(split_ratio[1] * n)
        test_size = n - train_size - val_size
        train_ids.extend(species_df_index[:train_size])
        val_ids.extend(species_df_index[train_size:train_size + val_size])
        test_ids.extend(species_df_index[train_size + val_size:])

    train_df = df.loc[train_ids]
    val_df = df.loc[val_ids]
    test_df = df.loc[test_ids]

    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    train_df.to_csv(path + "train.csv", index=False)
    val_df.to_csv(path + "val.csv", index=False)
    test_df.to_csv(path + "test.csv", index=False)

    print(f"DataFrames saved to {path}")

    return train_df, val_df, test_df
