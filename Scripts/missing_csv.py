import os
import pandas as pd
from pydub import AudioSegment
import unicodedata

# Base directories for audio files and CSV files
audio_files_base_directory = '/Users/aveliyath/PyScripts/Xeno-Canto_Data/Audio_Files/DUMP/Audio'
csv_files_base_directory = '/Users/aveliyath/PyScripts/Xeno-Canto_Data/Audio_Files/DUMP/CSV'

# Function to format duration from milliseconds to MM:SS format.
def format_duration(duration_ms):
    minutes, seconds = divmod(duration_ms // 1000, 60)
    return f"{int(minutes):02}:{int(seconds):02}"

# Normalize text to NFC form for consistent comparison
def normalize_text(text):
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize('NFC', text).strip()
    return text

# Add missing CSV entries for audio files without an entry
def add_missing_csv_entries(audio_base_dir, csv_base_dir):
    for root, dirs, files in os.walk(audio_base_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                audio_file_path = os.path.join(root, file)
                bird_species = os.path.basename(os.path.dirname(audio_file_path))
                csv_file_path = os.path.join(csv_base_dir, f"{bird_species}_metadata.csv")

                # Load existing CSV or create new if doesn't exist
                if os.path.exists(csv_file_path):
                    df = pd.read_csv(csv_file_path, dtype={'file-name': str})
                    df['file-name'] = df['file-name'].apply(normalize_text)
                else:
                    df = pd.DataFrame(columns=['file-name', 'length'])

                # Check if audio file already has an entry in CSV
                normalized_file_name = normalize_text(file)
                if not df[df['file-name'] == normalized_file_name].empty:
                    print(f"Entry already exists for {normalized_file_name} in {bird_species}_metadata.csv")
                    continue

                audio = AudioSegment.from_file(audio_file_path)
                new_row = {'file-name': normalized_file_name, 'length': format_duration(len(audio))}
                # Append the new row to the dataframe
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"Added entry for {normalized_file_name} in {bird_species}_metadata.csv")

                # Save the updated CSV
                df.to_csv(csv_file_path, index=False, encoding='utf-8')

if __name__ == "__main__":
    add_missing_csv_entries(audio_files_base_directory, csv_files_base_directory)
