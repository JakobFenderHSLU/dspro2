import argparse

from src.util.DataSplitUtil import split_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to the directory containing the data files")
    parser.add_argument("-r", "--ratios", type=str, nargs="+", help="ratios for train, val, test")
    parser.add_argument("-s", "--seed", type=int, help="seed value for random number generator")
    parser.add_argument("-y", "--yes", action="store_true", help="yes to skip confirmation prompt")

    args = parser.parse_args()

    if args.path is None:
        args.path = "./input/scrape/"
    if args.ratios is None:
        args.ratios = [0.6, 0.2, 0.2]
    else:
        args.ratios = [float(r) for r in args.ratios]
    if args.seed is None:
        args.seed = 42

    print("Arguments:")
    print(f"Path: {args.path}")
    print(f"Ratios: {args.ratios}")
    print(f"Seed: {args.seed}")

    # confirm before proceeding
    if args.yes is None:
        proceed = input("Proceed with splitting data? [Y/n]: ")
        if proceed.lower() == "n" or proceed.lower() == "no":
            print("Exiting...")
            exit()

    split_data(args.path, args.ratios, args.seed)
