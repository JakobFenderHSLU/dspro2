import requests
import os
import csv
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Scraper:
    def __init__(self, birds, quality_ratings, max_length_seconds, total_duration_cap, base_csv_dir, audio_files_base_dir):
        self.base_url = "https://xeno-canto.org/api/2/recordings"
        self.birds = birds
        self.quality_ratings = quality_ratings
        self.max_length_seconds = max_length_seconds
        self.total_duration_cap = total_duration_cap
        self.base_csv_dir = base_csv_dir
        self.audio_files_base_dir = audio_files_base_dir
        os.makedirs(base_csv_dir, exist_ok=True)
        os.makedirs(audio_files_base_dir, exist_ok=True)
        logging.info("Initialized BirdCall Scraper with configured settings.")

    # Parse the length of the recording from the string format
    def parse_length(self, length_str):
        try:
            parts = length_str.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                return 0
        except ValueError:
            logging.error(f'Error converting length due to format error: {length_str}')
            return 0

    # Scrape data for each bird species
    def scrape_data(self):
        logging.info("Starting data scraping for each bird species.")
        try:
            for bird in self.birds:
                self.scrape_bird_data(bird)
        except Exception as e:
            logging.error(f"An error occurred during data scraping: {e}")

    # Scrape data for a single bird species
    def scrape_bird_data(self, bird):
        query = bird.replace(" ", "+")
        url = f"{self.base_url}?query={query}"
        logging.info(f"Fetching data for {bird} from URL: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            logging.info(f"Data retrieval successful for {bird}. Processing recordings...")
            self.process_recordings(response.json(), bird)
        else:
            logging.error(f"Failed to retrieve data for {bird}. Status code: {response.status_code}")
    
    # Process the recordings for a single bird species
    def process_recordings(self, data, bird):
        try:
            bird_dir = os.path.join(self.audio_files_base_dir, bird.replace(" ", "_"))
            os.makedirs(bird_dir, exist_ok=True)
            csv_filename = os.path.join(self.base_csv_dir, f"{bird.replace(' ', '_')}_metadata.csv")
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['id', 'gen', 'sp', 'en', 'cnt', 'loc', 'lat', 'lng', 'type', 'sex', 'stage', 'url', 'file-name', 'quality', 'length', 'time', 'date', 'uploaded']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                total_duration = 0
                logging.info(f"Processing recordings for {bird}.")
                for recording in data['recordings']:
                    total_duration, cap_reached = self.process_recording(recording, bird_dir, writer, total_duration, bird)
                    if cap_reached:
                        logging.info(f"Duration cap reached for {bird}, stopping further downloads.")
                        break
        except Exception as e:
            logging.error(f"An error occurred while processing recordings for {bird}: {e}")

    # Process a single recording
    def process_recording(self, recording, bird_dir, writer, total_duration, bird):
        length_seconds = self.parse_length(recording.get('length', '0:00'))
        if recording['q'] in self.quality_ratings and length_seconds <= self.max_length_seconds:
            if total_duration + length_seconds > self.total_duration_cap:
                return total_duration, True
            total_duration += length_seconds
            self.download_recording(recording, bird_dir, writer)
        return total_duration, False

    # Download recording
    def download_recording(self, recording, bird_dir, writer):
        audio_url = "https:" + recording['file'] if not recording['file'].startswith("https:") else recording['file']
        logging.info(f"Attempting to download recording from {audio_url}")
        audio_response = requests.get(audio_url, timeout=10)
        if audio_response.status_code == 200:
            file_path = os.path.join(bird_dir, recording['file-name'])
            with open(file_path, "wb") as file:
                file.write(audio_response.content)
            logging.info(f"Successfully downloaded and saved recording to {file_path}")
            metadata = {k: recording.get(k, '') for k in writer.fieldnames}
            metadata['quality'] = recording['q']
            writer.writerow(metadata)
        else:
            logging.error(f"Failed to download audio for recording ID {recording['id']}. HTTP Status Code: {audio_response.status_code}")
