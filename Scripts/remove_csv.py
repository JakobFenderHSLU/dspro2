import os
import pandas as pd
import unicodedata

# Define the base directories for audio files and CSV files
audio_files_base_dir = '/Users/aveliyath/PyScripts/DUMP/Audio-DUMP'
csv_files_base_dir = '/Users/aveliyath/PyScripts/Xeno-Canto_Data/CSV'

def normalize_text(text):
    return unicodedata.normalize('NFC', text) if isinstance(text, str) else text

def remove_csv_entry_for_audio_file(audio_file_path, csv_directory):
    try:
        # Extract bird species from the audio file path
        bird_species = os.path.basename(os.path.dirname(audio_file_path))
        csv_file_path = os.path.join(csv_directory, f"{bird_species}_metadata.csv")
        
        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path)
            df['file-name'] = df['file-name'].apply(normalize_text)
            audio_file_name = os.path.basename(audio_file_path)
            normalized_audio_file_name = normalize_text(audio_file_name)
            
            # Check if the audio file's entry exists in the CSV
            if normalized_audio_file_name in df['file-name'].values:
                # Remove the entry from the DataFrame
                df = df[df['file-name'] != normalized_audio_file_name]
                df.to_csv(csv_file_path, index=False)
                print(f"Removed entry for {audio_file_name} from {csv_file_path}")
            else:
                # Highlight that no entry was found for the audio file
                print(f"\033[1;31mNo entry found for {audio_file_name} in {csv_file_path}\033[0m")
        else:
            print(f"CSV file for {bird_species} does not exist.")
    except Exception as e:
        print(f"Error while processing {audio_file_path}: {e}")

# Iterate through each bird species directory in the audio files directory
for root, dirs, files in os.walk(audio_files_base_dir):
    for file in files:
        if file.lower().endswith('.mp3'):
            audio_file_path = os.path.join(root, file)
            print(f'Processing audio file {audio_file_path}...')
            remove_csv_entry_for_audio_file(audio_file_path, csv_files_base_dir)
