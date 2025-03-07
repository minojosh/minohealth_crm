import os
from pathlib import Path
import requests
from tenacity import retry, stop_after_attempt, wait_fixed


# Define the root directory
root_dir = os.path.abspath(os.path.dirname(__file__))

# Create full path to database file
DATABASE_FILE = os.path.join(root_dir, 'healthcare.db')

# Ensure the database directory exists
os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)

DATABASE_URL = f"sqlite:///{DATABASE_FILE}"
LLM_API_URL = "http://46.101.91.139:5000/inference"

import requests
from tenacity import retry, stop_after_attempt, wait_fixed
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def moremi(question: str, system_prompt=None) -> str:
    """Makes API call to generate sequence"""
    url = "http://46.101.91.139:5001/inference"
    data = {
        "query": question,
        "temperature": 1.0,
        "max_new_token": 50,
        "systemPrompt": system_prompt
    }
    response = requests.post(url, json=data)
    return response.text.strip()