import argparse
import os
import pathlib

import pandas as pd
import torch

from src.basemodel.runner import BasemodelRunner
from src.util.FileUtils import validate

POSSIBLE_MODELS = ["cnn", "cnn-transfer"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to the data file")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
    parser.add_argument("-s", "--skip-validation", action="store_true", help="skip validation")
    parser.add_argument('-m', '--model', default=POSSIBLE_MODELS[0], const=POSSIBLE_MODELS[0],
                        choices=POSSIBLE_MODELS, help="model to use for training", nargs='?')
    parser.add_argument("-c", "--cpu", action="store_true", help="use CPU instead of GPU")
    parser.add_argument("-d", "--debug", action="store_true", help="debug mode")

    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.cpu:
        print("CUDA not available. Please make sure you have a CUDA-enabled GPU. "
              "If you want to train on CPU, use the --cpu flag.")
        exit()

    if args.path is None:
        args.path = "./input/scrape/"

    path = pathlib.Path(args.path)

    if not args.skip_validation:
        validate(path, verbose=args.verbose)

    if path.is_dir():
        train_df = pd.read_csv(path / "train.csv")
        test_df = pd.read_csv(path / "val.csv")

        # shuffle
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        test_df = test_df.sample(frac=1).reset_index(drop=True)

        if args.debug:
            # most samples per species in the first 10 species

            # get id of top 3 species with most samples
            most_samples_train = train_df[train_df['species_id'] < 10]['species_id'].value_counts().head(2)
            most_samples_test = test_df[test_df['species_id'] < 10]['species_id'].value_counts().head(2)

            train_df = train_df[train_df['species_id'].isin(most_samples_train.index)]
            test_df = test_df[test_df['species_id'].isin(most_samples_test.index)]

            # replace old index with new
            train_df['species_id'] = train_df['species_id'].replace(most_samples_train.index, [0, 1])
            test_df['species_id'] = test_df['species_id'].replace(most_samples_test.index, [0, 1])

            # reset index
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)

            # Set environment variables for debugging
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Synchronizes CPU and GPU
            os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Use CUDA Device-Side Assertions
        else:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            os.environ['TORCH_USE_CUDA_DSA'] = '0'

        if args.model == "cnn":
            print("Training base cnn model...")
            runner = BasemodelRunner(train_df, test_df)
            runner.run()
            print("Training complete!")

        elif args.model == "cnn-transfer":
            print("Training transfer learning model...")
