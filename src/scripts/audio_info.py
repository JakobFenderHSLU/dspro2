import os
from pydub import AudioSegment
audio_files_directory = '/Users/aveliyath/PyScripts/Xeno-Canto_Data/Audio_Files/Batch-4'
birds_count_low = []

def format_duration(duration_ms):
    hours, remainder = divmod(duration_ms, 3600000)
    minutes, seconds = divmod(remainder, 60000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds)//1000:02}"

def process_audio_folders(base_directory):
    for root, dirs, files in os.walk(base_directory):
        total_length_ms = 0
        audio_file_count = 0
        for file in files:
            if file.lower().endswith('.mp3'):
                try:
                    audio_file_count += 1
                except Exception as e:
                    print(f"Error counting files {file}: {e}")
        
        print(f"Directory: {root}")
        print(f"Number of audio files: {audio_file_count}")
        
        if audio_file_count < 5:
            birds_count_low.append(root)

if __name__ == "__main__":
    process_audio_folders(audio_files_directory)
    print("****************************************************************")
    for folder in birds_count_low:
        print(folder)
