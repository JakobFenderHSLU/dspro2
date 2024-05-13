import os
import pandas as pd
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import unicodedata

# Define the base directories for audio files and CSV files
audio_files_base_dir = '/Users/aveliyath/PyScripts/Xeno-Canto_Data/Audio_Files/Batch'
csv_files_base_dir = '/Users/aveliyath/PyScripts/Xeno-Canto_Data/CSV'

def convert_wav_to_mp3_and_remove_wav(wav_path):
    try:
        mp3_path = wav_path.replace('.wav', '.mp3').replace('.WAV', '.mp3')
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format='mp3', bitrate='320k')
        os.remove(wav_path)
    except CouldntDecodeError as e:
        print(f"Could not convert {wav_path} due to CouldntDecodeError: {e} !!!")
        remove_unsupported_files_and_csv_entries(wav_path, csv_files_base_dir)
    except Exception as e:
        print(f"Could not convert {wav_path} due to an unexpected error: {e} !!!")
        remove_unsupported_files_and_csv_entries(wav_path, csv_files_base_dir)

def rename_mp3_to_lowercase(file_path):
    try:
        new_file_path = file_path.replace('.MP3', '.mp3')
        os.rename(file_path, new_file_path)
    except Exception as e:
        print(f"Could not rename {file_path} due to error: {e} !!!")

def normalize_text(text):
    return unicodedata.normalize('NFC', text) if isinstance(text, str) else text

def remove_unsupported_files_and_csv_entries(file_path, csv_directory):
    try:
        os.remove(file_path)
        print(f'Removed unsupported or problematic file {file_path}')
        bird_species = os.path.basename(os.path.dirname(file_path))
        csv_file_path = os.path.join(csv_directory, f"{bird_species}_metadata.csv")
        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path)
            df['file-name'] = df['file-name'].apply(normalize_text)
            file_name = os.path.basename(file_path)
            df = df[df['file-name'] != normalize_text(file_name)]
            df.to_csv(csv_file_path, index=False, encoding='utf-8')
            print(f"Updated CSV by removing entry for {file_name}")
    except Exception as e:
        print(f"Error cleaning up for {file_path}: {e} !!!")

# Process each file in the audio files directory
for root, dirs, files in os.walk(audio_files_base_dir):
    for file in files:
        if file.lower().endswith(('.wav', '.WAV')):
            wav_path = os.path.join(root, file)
            print(f'Processing WAV file {wav_path}...')
            convert_wav_to_mp3_and_remove_wav(wav_path)
        elif file.upper().endswith('.MP3'):
            mp3_path = os.path.join(root, file)
            print(f'Processing MP3 file {mp3_path}...')
            rename_mp3_to_lowercase(mp3_path)
        elif not file.lower().endswith('.mp3'):
            # If the file is not an MP3 or WAV, remove it and its CSV entry
            file_path = os.path.join(root, file)
            print(f'Processing unsupported file {file_path}...')
            remove_unsupported_files_and_csv_entries(file_path, csv_files_base_dir)

# Update CSV file names
for csv_file in os.listdir(csv_files_base_dir):
    if csv_file.endswith('.csv'):
        csv_path = os.path.join(csv_files_base_dir, csv_file)
        try:
            df = pd.read_csv(csv_path)
            df['file-name'] = df['file-name'].apply(lambda x: x.replace('.wav', '.mp3').replace('.WAV', '.mp3').replace('.MP3', '.mp3') if isinstance(x, str) else x)
            df.to_csv(csv_path, index=False)
            print(f"Updated file names in CSV: {csv_file}")
        except pd.errors.EmptyDataError:
            print(f"CSV file {csv_file} is empty or not properly formatted !!!")
