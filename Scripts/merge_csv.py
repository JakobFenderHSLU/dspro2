import os
import pandas as pd

# Directory containing the CSV files
directory = '/Users/aveliyath/PyScripts/Xeno-Canto_Data/CSV_SUM'

# List to hold dataframes
dfs = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)

# Concatenate all dataframes in the list
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged dataframe to a new CSV file
output_file = os.path.join(directory, "merged_data.csv")
merged_df.to_csv(output_file, index=False)

print(f"Merged CSV saved to {output_file}")

