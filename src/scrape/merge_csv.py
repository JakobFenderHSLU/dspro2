import os
import pandas as pd
import logging

class MergeCSV:
    def __init__(self, csv_directory):
        self.csv_directory = csv_directory
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def merge_csv_files(self):
        dfs = []
        logging.info(f"Starting to merge CSV files in {self.csv_directory}")

        # Loop through all files in the directory
        for filename in os.listdir(self.csv_directory):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.csv_directory, filename)
                df = pd.read_csv(file_path)
                dfs.append(df)
                logging.info(f"Loaded CSV file {file_path}")

        # Concatenate all dataframes in the list
        merged_df = pd.concat(dfs, ignore_index=True)
        output_file = os.path.join(self.csv_directory, "merged_csv_files.csv")
        merged_df.to_csv(output_file, index=False)

        logging.info(f"Merged CSV saved to {output_file}")