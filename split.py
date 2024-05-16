import argparse

from src.util.DataSplitUtil import split_data
from src.util.LoggerUtils import init_logging

log = init_logging("split")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to the directory containing the data files")
    parser.add_argument("-r", "--ratios", type=str, nargs="+", help="ratios for train, val, test")
    parser.add_argument("-s", "--seed", type=int, help="seed value for random number generator")
    parser.add_argument("-y", "--yes", action="store_true", help="yes to skip confirmation prompt")
    parser.add_argument("-m", "--min", type=int, help="minimum number of samples per class")

    args = parser.parse_args()

    if args.path is None:
        args.path = "./input/scrape/"
    if args.ratios is None:
        args.ratios = [0.6, 0.2, 0.2]
    else:
        args.ratios = [float(r) for r in args.ratios]
    if args.seed is None:
        args.seed = 42

    if args.min is None:
        args.min = 0

    log.info("Arguments:")
    log.info(f"Path: {args.path}")
    log.info(f"Ratios: {args.ratios}")
    log.info(f"Seed: {args.seed}")
    log.info(f"Minimum samples per class: {args.min}")

    # confirm before proceeding
    if args.yes is None:
        proceed = input("Proceed with splitting data? [Y/n]: ")
        if proceed.lower() == "n" or proceed.lower() == "no":
            log.info("Exiting...")
            exit()

    split_data(args.path, args.ratios, args.seed, args.min)
