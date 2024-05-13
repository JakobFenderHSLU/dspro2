import argparse

from src.scrape.example import example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--full", type=str, help="ignore saved data and scrape all data")

    args = parser.parse_args()

    print(args.full)

    example()
