import os
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(path: str = "./../../input/scrape/", split_ratio: List[float] = [0.6, 0.2, 0.2], seed: int = 42,
               min_samples: int = 0):
    """
    Split the data into training, validation, and test sets based on the provided split ratio. Save the resulting
    DataFrames as CSV files in the specified directory.
    :param path: The path to the directory containing the data files.
    :param split_ratio: The ratio in which to split the data. Default is [0.6, 0.2, 0.2].
    :param seed: The seed value for the random number generator. Default is 42.
    :param min_samples: The minimum number of samples per class. Default is 0.
    :return: A DataFrame containing the split data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory {path} not found")

    folders = os.listdir(path)
    folders = [f for f in folders if os.path.isdir(os.path.join(path, f))]

    df_dict = {"species_name": [], "file_path": []}

    for folder in folders:
        files = os.listdir(os.path.join(path, folder))
        files = [f for f in files if f.endswith('.mp3')]

        if len(files) == 0:
            print(f"Warning: No files found in {folder}")

        for file in files:
            file_path = str(os.path.join(path, folder, file))
            df_dict["species_name"].append(folder)
            df_dict["file_path"].append(file_path)

    df = pd.DataFrame(df_dict)
    print("")

    # remove species with less than 3 samples
    species_counts = df["species_name"].value_counts()
    species_ids = species_counts[species_counts >= min_samples].index
    df = df[df["species_name"].isin(species_ids)]

    # filtered species
    filtered_out_species = species_counts[species_counts < min_samples].index
    for species in filtered_out_species:
        print(f"Filtered out species {species} with {species_counts[species]} samples")

    # one hot encode species_id as 1, 0
    species_dummies = pd.get_dummies(df["species_name"], prefix="species_name")
    df = pd.concat([df, species_dummies], axis=1)

    if split_ratio is None:
        raise ValueError("Split ratio must be provided")

    if len(split_ratio) != 3:
        raise ValueError("Split ratio must contain 3 values")
    if sum(split_ratio) != 1:
        raise ValueError("Split ratio values must sum to 1")

    X = df["file_path"]
    y = df[[col for col in df.columns if "species_name_" in col]]

    # make bool to int
    y = y.astype(int)

    # train val test split
    train_x, test_x, train_y, test_y = train_test_split(X, y,
                                                        test_size=sum(split_ratio[1:]),
                                                        random_state=seed,
                                                        stratify=y)
    val_x, test_x, val_y, test_y = train_test_split(test_x, test_y,
                                                    test_size=split_ratio[1] / (split_ratio[1] + split_ratio[2]),
                                                    random_state=seed, stratify=test_y)

    train_y = train_y.astype(int)

    val_y = val_y.astype(int)
    test_y = test_y.astype(int)

    # concatenate the X and y for train, val, test
    train_df = pd.concat([train_x, train_y], axis=1)
    val_df = pd.concat([val_x, val_y], axis=1)
    test_df = pd.concat([test_x, test_y], axis=1)

    print(f"Training set: {train_df.shape} samples")
    print(f"Validation set: {val_df.shape} samples")
    print(f"Test set: {test_df.shape} samples")

    train_df.to_csv(path + "train.csv", index=False)
    val_df.to_csv(path + "val.csv", index=False)
    test_df.to_csv(path + "test.csv", index=False)

    print(f"DataFrames saved to {path}")

    return train_df, val_df, test_df
