import argparse

from src.scrape.example import example
from src.util.LoggerUtils import init_logging

log = init_logging("scrape")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--full", type=str, help="ignore saved data and scrape all data")

    args = parser.parse_args()

    log.info(args.full)

    example()
