import requests
import os
import csv
import time

# Define the base URL of the Xeno-canto API
base_url = "https://xeno-canto.org/api/2/recordings"

# Updated list of bird species to search for
birds = [
"Yellowhammer",
"Pine Bunting",
"Rock Bunting",
"Ortolan Bunting",
"Cirl Bunting",
"Little Bunting",
"Rustic Bunting",
"Black-headed Bunting",
"Common Reed Bunting",
"Song Sparrow",
"Common Yellowthroat"
]

# Quality ratings to include
quality_ratings = ["A", "B"]

# Max length in seconds for the audio files to download (10 minutes)
max_length_seconds = 600

# Set the base directory for saving the CSV files
base_csv_dir = '/Users/aveliyath/PyScripts/Xeno-Canto_Data/CSV'
os.makedirs(base_csv_dir, exist_ok=True)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Convert recording length from 'MM:SS' format to seconds, handling unexpected formats.
def parse_length(length_str):
    try:
        parts = length_str.split(':')
        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:  # Handling format like 'HH:MM:SS' if found
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            return 0  # Default to 0 if format is unexpected
    except ValueError:
        print(f'error due to format conversion error of {length_str}')
        return 0  # Handle any conversion errors by defaulting to 0 seconds

for bird in birds:
    query = bird.replace(" ", "+")
    url = f"{base_url}?query={query}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(f"Results for {bird}: {len(data['recordings'])} recordings found.")

            bird_dir = os.path.join(script_dir, bird.replace(" ", "_"))
            os.makedirs(bird_dir, exist_ok=True)
            
            csv_filename = os.path.join(base_csv_dir, f"{bird.replace(' ', '_')}_metadata.csv")
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['id', 'gen', 'sp', 'en', 'cnt', 'loc', 'lat', 'lng', 'type', 'sex', 'stage', 'url', 'file-name', 'quality', 'length', 'time', 'date', 'uploaded']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for recording in data['recordings']:
                    length_seconds = parse_length(recording.get('length', '0:00'))
                    if recording['q'] in quality_ratings and length_seconds <= max_length_seconds:
                        print(f"Downloading {recording['en']} from {recording['cnt']} with quality {recording['q']}")
                        audio_url = "https:" + recording['file'] if not recording['file'].startswith("https:") else recording['file']

                        audio_response = requests.get(audio_url, timeout=10)
                        if audio_response.status_code == 200:
                            file_path = os.path.join(bird_dir, recording['file-name'])
                            with open(file_path, "wb") as file:
                                file.write(audio_response.content)
                            print(f"Saved to {file_path}")

                            metadata = {k: recording.get(k, '') for k in fieldnames}
                            metadata['quality'] = recording['q']
                            writer.writerow(metadata)
                        else:
                            print(f"Failed to download audio for recording ID {recording['id']}. HTTP Status Code: {audio_response.status_code} !!!")
                    else:
                        if length_seconds <= max_length_seconds:
                            print(f"Skipped {recording['id']} due to length exceeding 10 minutes")
                        else:
                            print(f"Skipped {recording['id']} due to quality.")
                print(f"Metadata for {bird} saved to {csv_filename}")
            print("\n")
        else:
            print(f"Failed to retrieve data for {bird}. Status code: {response.status_code} !!!")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data for {bird}: {str(e)} !!!")
    time.sleep(1)  # Respect the API rate limit
