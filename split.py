import argparse

from src.util.data_split_utils import split_data
from src.util.logger_utils import init_logging

log = init_logging("split")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to the directory containing the data files")
    parser.add_argument("-r", "--ratios", type=str, nargs="+", help="ratios for train, val, test")
    parser.add_argument("-s", "--seed", type=int, help="seed value for random number generator")
    parser.add_argument("-y", "--yes", action="store_true", help="yes to skip confirmation prompt")
    parser.add_argument("-ms", "--min-samples", type=int, help="minimum number of samples per class")
    parser.add_argument("-md", "--min-duration", type=int, help="minimum duration of audio files")
    parser.add_argument("-b", "--balance", action="store_true", help="balance the data")

    args = parser.parse_args()

    if args.path is None:
        args.path = "./input/scrape/segmented/"
    if args.ratios is None:
        args.ratios = [0.6, 0.2, 0.2]
    else:
        args.ratios = [float(r) for r in args.ratios]
    if args.seed is None:
        args.seed = 42

    if args.min_samples is None:
        args.min_samples = 0

    if args.min_duration is None:
        args.min_duration = 0

    log.info("Arguments:")
    log.info(f"Path: {args.path}")
    log.info(f"Ratios: {args.ratios}")
    log.info(f"Seed: {args.seed}")
    log.info(f"Minimum samples per class: {args.min_samples}")
    log.info(f"Minimum duration: {args.min_duration}")
    log.info(f"Balance: {args.balance}")

    # confirm before proceeding
    if args.yes is None:
        log.critical(
            "no confirmation provided. If you want do delete corrupt files, provide the -y flag. Exiting...")
        exit()

    log.info("Starting data split...")

    split_data(args.path, args.ratios, args.seed, args.min_samples, args.min_duration, args.balance)
