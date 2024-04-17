import requests
import os
import csv

# Define the base URL of the Xeno-canto API
base_url = "https://xeno-canto.org/api/2/recordings"

# Updated list of bird species to search for
birds = ["Blyth's Reed Warbler"]

# Quality ratings to include
quality_ratings = ["A", "B"]

# Max length in seconds for the audio files to download (10 minutes)
max_length_seconds = 600

# Set the base directory for saving the CSV files
base_csv_dir = '/Users/aveliyath/PyScripts/Xeno-Canto-Test/CSV'
os.makedirs(base_csv_dir, exist_ok=True)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

for bird in birds:
    query = bird.replace(" ", "+")
    url = f"{base_url}?query={query}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Results for {bird}:")
        
        bird_dir = os.path.join(script_dir, bird.replace(" ", "_"))
        os.makedirs(bird_dir, exist_ok=True)
        
        csv_filename = os.path.join(base_csv_dir, f"{bird.replace(' ', '_')}_metadata.csv")
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['id', 'gen', 'sp', 'en', 'cnt', 'loc', 'lat', 'lng', 'type', 'sex', 'stage', 'url', 'file-name', 'quality', 'length', 'time', 'date', 'uploaded']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for recording in data['recordings']:
                length = int(recording.get('length', '0').split(':')[0]) * 60 + int(recording.get('length', '0').split(':')[1])
                if recording['q'] in quality_ratings and length <= max_length_seconds:
                    print(f"Downloading {recording['en']} from {recording['cnt']} with quality {recording['q']}")
                    audio_url = "https:" + recording['file'] if not recording['file'].startswith("https:") else recording['file']

                    try:
                        audio_response = requests.get(audio_url, timeout=10)

                        if audio_response.status_code == 200:
                            file_path = os.path.join(bird_dir, recording['file-name'])

                            try:
                                with open(file_path, "wb") as file:
                                    file.write(audio_response.content)
                                print(f"Saved to {file_path}")

                                metadata = {k: recording.get(k, '') for k in fieldnames}
                                metadata['quality'] = recording['q']
                                writer.writerow(metadata)
                            except Exception as e:
                                print(f"Failed to save or process audio for recording ID {recording['id']} due to error: {e} !!!")
                        else:
                            print(f"Failed to download audio for recording ID {recording['id']}. HTTP Status Code: {audio_response.status_code} !!!")
                    except requests.exceptions.RequestException as e:
                        print(f"Request failed for recording ID {recording['id']} with error: {e} !!!")
            print(f"Metadata for {bird} saved to {csv_filename}")
        print("\n")
    else:
        print(f"Failed to retrieve data for {bird}. Status code: {response.status_code} !!!")
