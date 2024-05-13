import argparse
import pathlib

import pandas as pd

from src.util.FileUtils import validate

POSSIBLE_MODELS = ["cnn", "cnn-transfer"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to the data file")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
    parser.add_argument("-s", "--skip-validation", action="store_true", help="skip validation")
    parser.add_argument('-m', '--model', default=POSSIBLE_MODELS[0], const=POSSIBLE_MODELS[0],
                        choices=POSSIBLE_MODELS, help="model to use for training")

    args = parser.parse_args()

    if args.path is None:
        args.path = "./input/scrape/"

    path = pathlib.Path(args.path)

    if not args.skip_validation:
        validate(path, verbose=args.verbose)

    if path.is_dir():
        train_df = pd.read_csv(path / "train.csv")
        test_df = pd.read_csv(path / "test.csv")

        if args.model == "cnn":
            print("Training CNN model...")
        elif args.model == "cnn-transfer":
            print("Training transfer learning model...")
