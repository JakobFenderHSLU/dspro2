import argparse

import pandas as pd

from src.scratch.tester import CnnFromScratchTester
from src.util.logger_utils import init_logging

log = init_logging("test")

POSSIBLE_MODELS = ["cnn", "knn", "vggish"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to the model file")
    parser.add_argument("-dp", "--data-path", type=str, help="path to the data file")
    parser.add_argument('-m', '--model', default=POSSIBLE_MODELS[0], const=POSSIBLE_MODELS[0],
                        choices=POSSIBLE_MODELS, help="model to use for training", nargs='?')

    args = parser.parse_args()

    log.info("Arguments:")
    log.info(f"Path: {args.path}")
    log.info(f"Model: {args.model}")

    test_df = pd.read_csv(args.data_path)

    assert args.path is not None, "Path is required"

    if args.model == "cnn":
        tester = CnnFromScratchTester(args.path, test_df)
        tester.test()
    else:
        log.warning("Your model is not supported yet")



