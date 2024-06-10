import argparse
import os
import pathlib
import time

import pandas as pd
import torch

from src.basemodel.runner import BasemodelRunner
from src.util.file_utils import validate
from src.util.logger_utils import init_logging
from src.util.scale_utils import convert_to_small, convert_to_debug

log = init_logging("train")

POSSIBLE_MODELS = ["cnn", "cnn-transfer"]
POSSIBLE_SCALES = ["full", "small", "debug"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to the data file")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
    parser.add_argument("-sv", "--skip-validation", action="store_true", help="skip validation")
    parser.add_argument('-m', '--model', default=POSSIBLE_MODELS[0], const=POSSIBLE_MODELS[0],
                        choices=POSSIBLE_MODELS, help="model to use for training", nargs='?')
    parser.add_argument("-c", "--cpu", action="store_true", help="use CPU instead of GPU")
    parser.add_argument("-s", "--scale", default=POSSIBLE_SCALES[0], const=POSSIBLE_SCALES[0],
                        choices=POSSIBLE_SCALES, help="scale of the dataset", nargs='?')

    script_start_time = time.time()

    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.cpu:
        log.critical("CUDA not available. Please make sure you have a CUDA-enabled GPU. "
                     "If you want to train on CPU, use the --cpu flag.")
        exit()

    if args.path is None:
        args.path = "./input/scrape/segmented/"

    path = pathlib.Path(args.path)

    if not args.skip_validation:
        validate(path, verbose=args.verbose)

    if path.is_dir():
        train_df = pd.read_csv(path / "train.csv")
        val_df = pd.read_csv(path / "val.csv")

        if args.scale == "full":
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            os.environ['TORCH_USE_CUDA_DSA'] = '0'

        elif args.scale == "small":
            # most samples per species in the first 7 species
            train_df, val_df = convert_to_small(train_df, val_df, 4)

            # Set environment variables for debugging
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Synchronizes CPU and GPU
            os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Use CUDA Device-Side Assertions

        elif args.scale == "debug":
            train_df, val_df = convert_to_debug(train_df, val_df)

        if args.model == "cnn":
            log.info("Training base cnn model...")
            runner = BasemodelRunner(train_df, val_df, args.scale)
            runner.run()

        elif args.model == "cnn-transfer":
            log.info("Training transfer learning model...")

    # HH:MM:SS
    formatted_duration = time.strftime("%H:%M:%S", time.gmtime(time.time() - script_start_time))
    log.info(f"Training complete! (Time taken: {formatted_duration})")
