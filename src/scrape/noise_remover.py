import logging
import noisereduce as nr
import librosa
import soundfile as sf
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NoiseRemover:
    def __init__(self, audio_files_base_dir):
        self.audio_files_base_dir = audio_files_base_dir

    # remove background noise from audio file by using noisereduce library
    def remove_background_noise(self, file_path):
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=None)
            if len(y) == 0:
                raise ValueError("File contains no audio data or has incompatible encoding.")

            # Perform noise reduction
            reduced_noise = nr.reduce_noise(y=y, sr=sr)

            # Overwrite the original file to maintain the name
            sf.write(file_path, reduced_noise, sr)
            logging.info(f"Processed and saved cleaned audio to {file_path}")
        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}")

    def remove_noise(self):
        try:
            for root, _, files in os.walk(self.audio_files_base_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.lower().endswith('.mp3'):
                        if os.path.getsize(file_path) > 0:  # Check if the file is not empty
                            self.remove_background_noise(file_path)
                        else:
                            logging.warning(f"Skipped empty or corrupt file {file_path}")
        except Exception as e:
            logging.error(f"An error occurred during file cleaning: {e}")
