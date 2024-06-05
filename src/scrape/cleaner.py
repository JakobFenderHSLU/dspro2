import os
import pandas as pd
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Cleaner:
    def __init__(self, audio_files_base_dir, csv_files_base_dir):
        self.audio_files_base_dir = audio_files_base_dir
        self.csv_files_base_dir = csv_files_base_dir

    # convert wav files to mp3 and remove the wav files
    def convert_wav_to_mp3_and_remove_wav(self, wav_path):
        try:
            mp3_path = wav_path.replace('.wav', '.mp3').replace('.WAV', '.mp3')
            audio = AudioSegment.from_wav(wav_path)
            audio.export(mp3_path, format='mp3', bitrate='320k')
            os.remove(wav_path)
            logging.info(f"Removed {wav_path} and converted it to MP3 in {mp3_path} successfully.")
        except CouldntDecodeError as e:
            logging.error(f"Could not convert {wav_path} due to CouldntDecodeError: {e}")
            self.remove_unsupported_files_and_csv_entries(wav_path)
        except Exception as e:
            logging.error(f"Could not convert {wav_path} due to an unexpected error: {e}")
            self.remove_unsupported_files_and_csv_entries(wav_path)

    # check if the file is valid by trying to load it with pydub
    def remove_unsupported_files_and_csv_entries(self, file_path):
        try:
            os.remove(file_path)
            logging.info(f'Removed unsupported or problematic file {file_path}')
            bird_species = os.path.basename(os.path.dirname(file_path))
            csv_file_path = os.path.join(self.csv_files_base_dir, f"{bird_species}_metadata.csv")
            if os.path.exists(csv_file_path):
                df = pd.read_csv(csv_file_path)
                df = df[df['file-name'] != os.path.basename(file_path)]
                df.to_csv(csv_file_path, index=False, encoding='utf-8')
                logging.info(f"Updated CSV by removing entry for {os.path.basename(file_path)}")
        except Exception as e:
            logging.error(f"Error cleaning up for {file_path}: {e}")
            
    def is_file_valid(self, file_path):
        try:
            audio = AudioSegment.from_file(file_path)
            return True
        except Exception as e:
            logging.error(f"File {file_path} is corrupted or invalid: {e}")
            return False

    def clean_files(self):
        try:
            for root, _, files in os.walk(self.audio_files_base_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.lower().endswith('.wav'):
                        self.convert_wav_to_mp3_and_remove_wav(file_path)
                    elif file.upper().endswith('.MP3'):
                        new_path = file_path[:-4] + '.mp3'
                        os.rename(file_path, new_path)
                        logging.info(f"Renamed {file_path} to {new_path} to standardize the file extension.")
                    elif not file.lower().endswith('.mp3'):
                        self.remove_unsupported_files_and_csv_entries(file_path)
        except Exception as e:
            logging.error(f"An error occurred during file cleaning: {e}")
