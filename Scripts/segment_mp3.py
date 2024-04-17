import os
import pandas as pd
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import unicodedata


# Define directories
AUDIO_BASE_DIR = '/Users/aveliyath/PyScripts/Xeno-Canto_Data/Audio_Files/Batch-6'
CSV_BASE_DIR = '/Users/aveliyath/PyScripts/Xeno-Canto_Data/CSV'
"""
# Define directories
AUDIO_BASE_DIR = '/Users/aveliyath/PyScripts/Xeno-Canto-Test/Audio_Files/Batch'
CSV_BASE_DIR = '/Users/aveliyath/PyScripts/Xeno-Canto-Test/CSV'
# Define directories
AUDIO_BASE_DIR = '/Users/aveliyath/PyScripts/Xeno-Canto_Data/Audio_Files/DUMP/Audio-DUMP'
CSV_BASE_DIR = '/Users/aveliyath/PyScripts/Xeno-Canto_Data/Audio_Files/DUMP/CSV'
"""

SEGMENT_LENGTH_MS = 15000  # Segment length in milliseconds

def normalize_text(text):
    # Ensure the input is a string before normalization
    if isinstance(text, str):
        return unicodedata.normalize('NFC', text)
    else:
        # Return the text as is if it's not a string
        return text

def format_length(duration_seconds):
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    return f"{minutes:02}:{seconds:02}"

def segment_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        duration_ms = len(audio)
        if duration_ms <= SEGMENT_LENGTH_MS:
            return None, None
        segments = [audio[start_ms:start_ms + SEGMENT_LENGTH_MS] for start_ms in range(0, duration_ms, SEGMENT_LENGTH_MS)]
        return segments, None
    except (CouldntDecodeError, IndexError) as e:
        print(f"Could not process file {file_path} due to an error: {e} !!!")
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"{file_path} has been deleted due to processing errors !!!")
        return None, e

def process_audio_files():
    for entry in os.scandir(AUDIO_BASE_DIR):
        if entry.is_dir():
            species_dir = entry.path
            bird_species = entry.name
            csv_path = os.path.join(CSV_BASE_DIR, f"{bird_species}_metadata.csv")
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
                # Apply normalization only to non-null and string values
                df['file-name'] = df['file-name'].apply(lambda x: normalize_text(x) if isinstance(x, str) else x)
            except pd.errors.EmptyDataError:
                print(f"CSV file for {species_dir} is empty or not found !!!")
                continue
            except Exception as e:
                print(f"CSV file processing error for {species_dir} with {e} !!!")


            df = process_files_in_directory(species_dir, bird_species, df, csv_path)
            if not df.empty:
                df.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"Finished processing files in {species_dir}. CSV updated.")

def process_files_in_directory(species_dir, bird_species, df, csv_path):
    for file_entry in os.scandir(species_dir):
        if file_entry.is_file() and file_entry.name.lower().endswith('.mp3'):
            file_path = file_entry.path
            print(f"Processing {file_entry.name} in {file_entry.path}...")
            segments, error = segment_audio(file_path)
            if segments:
                df = update_csv(df, file_entry.name, segments, species_dir)
            elif error:
                print(f"Error processing {file_entry.name}: {error} !!!")
    return df

def update_csv(df, original_file_name, segments, species_dir):
    normalized_original_file_name = normalize_text(original_file_name)  # Normalize including extension
    condition = df['file-name'] == normalized_original_file_name  # Exact match condition
    if not df[condition].empty:
        df = df.drop(df[condition].index)  # Drop the original entry if found
    
    # Proceed to append new rows for segments and export them
    new_rows = [{'file-name': f"{normalize_text(original_file_name).rsplit('.', 1)[0]}_{i}.mp3", 'length': format_length(len(segment) / 1000)} for i, segment in enumerate(segments, start=1)]
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    
    for i, segment in enumerate(segments, start=1):
        segment_name = f"{normalize_text(original_file_name).rsplit('.', 1)[0]}_{i}.mp3"
        segment_path = f"{species_dir}/{segment_name}"
        segment.export(segment_path, format='mp3')
    
    original_file_path = f"{species_dir}/{original_file_name}"
    if os.path.exists(original_file_path):
        os.remove(original_file_path)
        print(f"Original file {original_file_name} removed after segmentation.")
    
    return df


if __name__ == "__main__":
    process_audio_files()
