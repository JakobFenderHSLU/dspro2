import os
import pandas as pd
import unicodedata

# Base directories for audio files and CSV files
AUDIO_BASE_DIR = '/Users/aveliyath/PyScripts/Xeno-Canto_Data/Audio_Files/Batch-5'
CSV_BASE_DIR = '/Users/aveliyath/PyScripts/Xeno-Canto_Data/CSV'

def normalize_text(text):
    if isinstance(text, str):
        return unicodedata.normalize('NFC', text)
    return text

def delete_excess_segments(audio_base_dir, csv_base_dir, max_segments=40):
    for root, dirs, files in os.walk(audio_base_dir):
        for file in sorted(files):
            if file.lower().endswith('.mp3'):
                parts = file.split('_')
                try:
                    # Attempt to parse the segment index
                    segment_index = int(parts[-1].split('.')[0])
                    base_name = '_'.join(parts[:-1])
                    
                    if segment_index > max_segments:
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
                        print(f"Removed excess segment {file_path}")
                        # Optionally, remove from CSV
                        remove_csv_entry_if_needed(root, file, csv_base_dir)
                except Exception as e:
                    # Handle files that do not conform to the expected naming convention
                    print(f"For {file} following exception {e} has occurred !!!")

def remove_csv_entry_if_needed(root, file_name, csv_base_dir):
    bird_species = os.path.basename(root)
    csv_file_path = os.path.join(csv_base_dir, f"{bird_species}_metadata.csv")
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        df['file-name'] = df['file-name'].apply(normalize_text)
        normalized_file_name = normalize_text(file_name)
        
        if normalized_file_name in df['file-name'].values:
            df = df[df['file-name'] != normalized_file_name]
            df.to_csv(csv_file_path, index=False)
            print(f"Updated CSV after removing {file_name} from {csv_file_path}")

if __name__ == "__main__":
    delete_excess_segments(AUDIO_BASE_DIR, CSV_BASE_DIR)
