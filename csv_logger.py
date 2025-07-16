# csv_logger.py
import csv
import os
from datetime import datetime

LOG_FILE = 'logs/voice_logs.csv'
os.makedirs('logs', exist_ok=True)

def log_to_csv(name, user_id, action, file_path=''):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Name', 'User ID', 'Action', 'Timestamp', 'File Path'])
        writer.writerow([name, user_id, action, timestamp, file_path])
