import argparse
import pathlib

import pandas as pd
import torch

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

    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.cpu:
        print("CUDA not available. Please make sure you have a CUDA-enabled GPU. "
              "If you want to train on CPU, use the -c flag.")
        exit()

    if args.path is None:
        args.path = "./input/scrape/"

    path = pathlib.Path(args.path)

    if not args.skip_validation:
        validate(path, verbose=args.verbose)

    if path.is_dir():
        train_df = pd.read_csv(path / "train.csv")
        test_df = pd.read_csv(path / "val.csv")

        if args.model == "cnn":
            print("Training base cnn model...")
        elif args.model == "cnn-transfer":
            print("Training transfer learning model...")
