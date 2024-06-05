from src.scrape.scraper import Scraper
from src.scrape.cleaner import Cleaner
from src.scrape.noise_remover import NoiseRemover
from src.scrape.merge_csv import MergeCSV
from src.util.LoggerUtils import init_logging
from src.scrape.bird_species import species_list

import argparse
import shutil
import os

log = init_logging("start of the process")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--full", type=str, help="ignore saved data and scrape all data")
    args = parser.parse_args()

    birds = species_list
    quality_ratings = ["A", "B", "C"]
    max_length_seconds = 600
    total_duration_cap = 11250
    base_csv_dir = 'input/scrape/csv'
    audio_files_base_dir = 'input/scrape/input_Removal'
    audio_files_no_removal_dir = 'input/scrape/input_noRemoval'

    # Ensure directories exist
    for directory in [base_csv_dir, audio_files_base_dir, audio_files_no_removal_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    # Scrape data
    if args.full:
        log.info("Starting full data scrape.")
        scraper = Scraper(birds, quality_ratings, max_length_seconds, total_duration_cap, base_csv_dir, audio_files_no_removal_dir)
        scraper.scrape_data()
    
    # Clean data
    log.info("Starting cleaning files.")
    cleaner = Cleaner(audio_files_no_removal_dir, base_csv_dir)
    cleaner.clean_files()

    # Copy cleaned files to the noise removal directory
    for filename in os.listdir(audio_files_no_removal_dir):
        src_path = os.path.join(audio_files_no_removal_dir, filename)
        dest_path = os.path.join(audio_files_base_dir, filename)
        shutil.copy(src_path, dest_path)  
        log.info(f"Copied {src_path} to {dest_path} for noise removal.")

    # Remove background noise
    log.info("Starting noise removal.")
    noise_remover = NoiseRemover(audio_files_base_dir)
    noise_remover.remove_noise()

    # Merge CSV files
    csv_merger = MergeCSV(base_csv_dir)
    csv_merger.merge_csv_files()

if __name__ == "__main__":
    main()
