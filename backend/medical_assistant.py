"""
Moremi Speech Assistant - Production Version
A speech-based medical assistant that uses an STT server for speech-to-text
and text-to-speech functionality, integrated with the Moremi AI model.
"""

import os
import json
import time
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import sys
import select
import termios
import tty

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from .database import Patient, Session, get_session
from .moremi import ConversationManager
# from .database_utils import PatientDatabaseManager
from .TTS_client import TTSClient
from .STT_client import SpeechRecognitionClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('speech_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# try:
#     from .database_utils import PatientDatabaseManager
# except ModuleNotFoundError:
#     logger.warning("Could not import `PatientDatabaseManager` from `appointment_medication_scheduling`")

@dataclass
class Message:
    """Represents a conversation message between user and assistant."""
    def __init__(self, user_input: str, response: Optional[str] = None):
        self.user_input = user_input
        self.response = response

    def add_response(self, response: str) -> None:
        """Add assistant's response to the message."""
        self.response = response

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return asdict(self)

class SpeechAssistant:
    """Main speech assistant class that handles speech recognition."""
    
    def __init__(self):
        """Initialize the speech assistant with necessary configurations."""
        self._setup_directories()
        self._setup_logging()
        
        # Update config path to use project root
        config_path = Path(__file__).parent.parent / '.env'
        if not config_path.exists():
            logger.warning(f"Environment file not found at {config_path}, using environment variables")
        load_dotenv(config_path)
        
        # Initialize STT client settings
        self.stt_server_url = os.getenv("STT_SERVER_URL", "https://1bf3-34-142-255-155.ngrok-free.app/")
        
        # Initialize other components
        self.messages: List[Message] = []
        self.api = os.getenv("MOREMI_API_URL")  # Fix: Use correct env var name
        self._load_prompts()
        self.db_manager = PatientDatabaseManager()
        self.patient_id = None

    def _setup_logging(self):
        """Configure logging to file and console."""
        log_dir = Path(__file__).parent.parent / 'data' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_file = log_dir / f"app_{self._get_timestamp()}.log"
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        data_dir = Path(__file__).parent.parent / 'data'
        dirs = [
            'audio',
            'transcripts',
            'responses',
            'yaml_output/raw',
            'yaml_output/processed',
            'logs'
        ]
        for dir_name in dirs:
            dir_path = data_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)

    def _get_timestamp(self) -> str:
        """Generate a formatted timestamp."""
        return datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

    def recognize_speech(self) -> str:
        """Record and transcribe speech from microphone using STT client."""
        timestamp = self._get_timestamp()
        audio_path = Path(__file__).parent.parent / 'data' / 'audio' / f'recording_{timestamp}.wav'
        transcript_path = Path(__file__).parent.parent / 'data' / 'transcripts' / f'transcript_{timestamp}.txt'
        
        logger.info("Starting speech recognition using STT client")
        print("\nListening... Press 'f' to finish recording")
        
        try:
            # Initialize the STT client
            stt_client = SpeechRecognitionClient(server_url=self.stt_server_url)
            
            # Start speech recognition
            logger.info("Starting speech recognition")
            transcript = stt_client.recognize_speech()
            
            if transcript:
                # Save transcript to file
                transcript_path.write_text(transcript)
                logger.info(f"Transcript saved to {transcript_path}")
                print(f"\nFinal transcript: {transcript}")
                
                # We don't save audio files here as they're handled by the STT client
                # But we log the information
                logger.info("Audio recording handled by STT client")
                
                return transcript
            else:
                logger.warning("No transcription received from STT client")
                print("\nNo speech was transcribed. Please try again.")
                return ""
                
        except Exception as e:
            logger.error(f"Error in speech recognition: {str(e)}")
            print(f"\nError in speech recognition: {str(e)}")
            return ""

    def get_moremi_response(self, messages: List[Message], system_prompt: Optional[str] = None) -> str:
        """Get response from Moremi API."""
        if not self.api:
            raise ValueError("Moremi API URL not configured")

        try:
            headers = {
                "Authorization": f"Bearer {os.getenv('MOREMI_API_KEY')}",
                # "azureml-model-deployment": os.getenv('MOREMI_DEPLOYMENT', 'llava-deployment'),
                "Content-Type": "application/json"
            }
            
            # Structure the context more clearly
            current_message = messages[-1] if messages else None
            if not current_message:
                raise ValueError("No message to process")

            # Create example YAML
            example_yaml = """
name: John Smith
dob: 1980-01-15
address: 123 Main Street
phone: 024-5678-123
insurance: ABC Insurance
condition: flu
symptoms:
  - cough
  - fever
reason_for_visit: Medical consultation
appointment_details:
  type: follow-up visit
  time: "09:30"
  doctor: Dr. Smith
  scheduled_date: "2025-01-15"
"""
            
            # Prepare the request data with clear context and query
            data = {
                'context': {
                    'transcript': current_message.user_input,
                    'timestamp': self._get_timestamp(),
                    'conversation_history': [
                        {
                            'user_input': msg.user_input,
                            'response': msg.response
                        } for msg in messages[:-1]
                    ] if len(messages) > 1 else []
                },
                'system_prompt': system_prompt or self.system_prompt,
                'query': (
                    f"Extract medical information from this transcript:\n\n"
                    f"{current_message.user_input}\n\n"
                    f"Format the extracted information as YAML, following this example structure:\n\n"
                    f"{example_yaml}\n\n"
                    "Remember:\n"
                    "1. Only include information explicitly stated in the transcript\n"
                    "2. Use the exact same field names as the example\n"
                    "3. For missing information, use None\n"
                    "4. Ensure the output is valid YAML\n"
                    "5. Do not include any other text, ONLY the YAML output"
                ),
                'temperature': 0.2,
                'max_tokens': 500
            }

            response = self._make_api_request(data, headers)
            
            if not response.ok:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
            # Parse response
            try:
                response_data = response.json()
                # Extract YAML content from response
                if isinstance(response_data, dict):
                    extracted_info = response_data.get('response', response_data)
                else:
                    extracted_info = response_data
                
                # Clean up YAML content
                if isinstance(extracted_info, str):
                    extracted_info = (extracted_info
                        .strip()
                        .replace('```yaml', '')
                        .replace('```', '')
                        .strip())
                
                # Validate YAML
                import yaml
                parsed_yaml = yaml.safe_load(extracted_info)
                if not isinstance(parsed_yaml, dict):
                    raise ValueError(f"Invalid YAML structure: {extracted_info}")
                
                # Convert back to string
                extracted_info = yaml.dump(parsed_yaml, default_flow_style=False)
                
                logger.debug(f"Parsed YAML: {extracted_info}")
                return extracted_info
                
            except Exception as e:
                logger.error(f"Failed to parse response: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Error getting Moremi response: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _make_api_request(self, data: Dict[str, Any], headers: Dict[str, str]) -> requests.Response:
        """Make API request with retry logic."""
        try:
            response = requests.post(self.api, json=data, headers=headers, timeout=30)
            logger.debug(f"API request status: {response.status_code}")
            return response
        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise

    def save_conversation(self) -> None:
        """Save the conversation history and process the transcription."""
        timestamp = self._get_timestamp()
        
        try:
            # Save raw conversation
            self.processor.save_raw_conversation(
                [msg.to_dict() for msg in self.messages],
                timestamp
            )
            
            # Process and save structured information
            extracted_info = self.processor.process_conversation(
                [msg.to_dict() for msg in self.messages],
                timestamp
            )
            
            if extracted_info:
                logger.info("Successfully extracted medical information")
            else:
                logger.warning("No medical information could be extracted")
                
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")

    def _load_prompts(self) -> None:
        """Load system prompts from configuration."""
        try:
            prompt_path = Path(__file__).parent.parent / 'data' / 'prompts' / 'prompt_config.json'
            if not prompt_path.exists():
                raise FileNotFoundError("Prompt configuration file not found")
            
            with open(prompt_path, 'r') as f:
                prompts = json.load(f)
            
            self.system_prompt = prompts.get('systemprompt', '')
            if not self.system_prompt:
                raise ValueError("System prompt not found in configuration")
            
            # Add default extraction guidelines if not present
            if 'extraction_guidelines' not in prompts:
                self.system_prompt += "\n\nExtraction Guidelines:\n" + \
                    "1. Only extract information explicitly mentioned in the transcript\n" + \
                    "2. Format output as valid YAML\n" + \
                    "3. Use consistent key names\n" + \
                    "4. Include timestamps where relevant\n" + \
                    "5. Mark uncertain information with 'confidence' field"
            
            logger.info("Prompts loaded successfully")
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            raise

    def _process_yaml(self, raw_yaml: str, timestamp: str) -> str:
        """Process and clean up YAML content."""
        try:
            import yaml
            from yaml import SafeDumper
            
            # Custom YAML dumper for better string formatting
            class MedicalYAMLDumper(SafeDumper):
                pass
            
            # Handle multiline strings better
            def str_presenter(dumper, data):
                if len(data.split('\n')) > 1:  # check for multiline string
                    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
                return dumper.represent_scalar('tag:yaml.org,2002:str', data)
            
            MedicalYAMLDumper.add_representer(str, str_presenter)
            
            # First, clean up any escape characters and normalize newlines
            raw_yaml = raw_yaml.replace('\\n', '\n').replace('\\_', '_')
            
            # Parse the YAML
            data = yaml.safe_load(raw_yaml)
            
            # If it's just a string (sometimes the API returns escaped string),
            # try to parse it again
            if isinstance(data, str):
                data = yaml.safe_load(data)
            
            if not isinstance(data, dict):
                raise ValueError("Invalid YAML structure")

            # Process the data but keep original values
            cleaned_data = {}
            
            # Standard fields - preserve original values but ensure proper structure
            fields = [
                'name', 'dob', 'address', 'phone', 'insurance', 
                'condition', 'reason_for_visit', "email"
            ]
            for field in fields:
                cleaned_data[field] = data.get(field)
            
            # Handle symptoms list
            symptoms = data.get('symptoms', [])
            if symptoms:
                if isinstance(symptoms, str):
                    # If symptoms came as string, split and clean
                    cleaned_data['symptoms'] = [s.strip() for s in symptoms.split(',')]
                else:
                    cleaned_data['symptoms'] = symptoms
            else:
                cleaned_data['symptoms'] = None
            
            # Process appointment details while preserving original values
            appt = data.get('appointment_details')
            if appt and isinstance(appt, dict):
                cleaned_data['appointment_details'] = {
                    'type': appt.get('type'),
                    'time': appt.get('time'),
                    'doctor': appt.get('doctor'),
                    'scheduled_date': appt.get('scheduled_date')
                }
            else:
                cleaned_data['appointment_details'] = None
            
            # Add metadata at the end
            cleaned_data['metadata'] = {
                'timestamp': timestamp,
                'processed_at': datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss"),
                'version': '1.0'
            }
            
            # Convert to YAML with proper formatting
            return yaml.dump(
                cleaned_data,
                Dumper=MedicalYAMLDumper,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
                indent=2,
                width=80
            )
            
        except Exception as e:
            logger.error(f"Error processing YAML: {str(e)}")
            return raw_yaml  # Return original if processing fails

    def _save_yaml(self, yaml_content: str, timestamp: str) -> Tuple[Path, Path]:
        """Save YAML content to both raw and processed directories."""
        base_dir = Path(__file__).parent.parent / 'data' / 'yaml_output'
        
        # Save raw YAML
        raw_path = base_dir / 'raw' / f'extracted_{timestamp}.yaml'
        raw_path.write_text(yaml_content)
        logger.info(f"Raw YAML saved to {raw_path}")
        
        # Process and save cleaned YAML
        processed_content = self._process_yaml(yaml_content, timestamp)
        processed_path = base_dir / 'processed' / f'extracted_{timestamp}.yaml'
        processed_path.write_text(processed_content)
        logger.info(f"Processed YAML saved to {processed_path}")
        
        return raw_path, processed_path

    def _create_patient_from_yaml(self, yaml_data: dict) -> Optional[int]:
        """Create a new patient from extracted YAML data."""
        try:
            logger.debug(f"Creating patient from YAML: {yaml_data}")
            
            # Extract patient data
            patient_data = {
                'name': yaml_data.get('name'),
                'phone': yaml_data.get('phone'),
                'email': yaml_data.get('email'),
                'address': yaml_data.get('address'),
                'date_of_birth': yaml_data.get('dob')
            }
            
            # Remove None values
            patient_data = {k: v for k, v in patient_data.items() if v is not None}
            
            if not patient_data.get('name'):
                logger.error("Missing required field: name")
                return None
                
            logger.debug(f"Attempting to create patient with data: {patient_data}")
            
            # Create patient
            patient_id = self.db_manager.create_patient(patient_data)
            
            if patient_id:
                logger.info(f"Created patient with ID: {patient_id}")
                
                # Handle appointment if present
                appt = yaml_data.get('appointment_details')
                if appt and isinstance(appt, dict):
                    logger.debug(f"Creating appointment with details: {appt}")
                    appt_id = self.db_manager.create_appointment(
                        patient_id=patient_id,
                        appointment_type=appt.get('type'),
                        scheduled_date=appt.get('scheduled_date'),
                        scheduled_time=appt.get('time'),
                        doctor_name=appt.get('doctor')
                    )
                    if appt_id:
                        logger.info(f"Created appointment {appt_id} for patient {patient_id}")
            
            return patient_id
            
        except Exception as e:
            logger.error(f"Failed to create patient: {str(e)}")
            return None

    def run(self, patient_id: Optional[int] = None) -> None:
        """
        Run the main conversation loop.
        
        Args:
            patient_id: Optional ID of the patient to update in the database
        """
        self.patient_id = patient_id
        print("\nMedical Transcription System")
        print("===========================")
        print("1. Start speaking when ready")
        print("2. Press 'f' to finish recording")
        print("3. The system will save:")
        print("   - Audio recording")
        print("   - Transcript")
        print("   - Extracted medical information (raw and processed)")
        print("===========================\n")
        
        try:
            # Get speech input
            transcript = self.recognize_speech()
            if not transcript:
                logger.warning("No speech detected")
                return
                
            # Process with Moremi
            self.messages.append(Message(user_input=transcript))
            response = self.get_moremi_response(self.messages, self.system_prompt)
            self.messages[-1].add_response(response)
            
            # Parse YAML response
            import yaml
            try:
                # Clean up the response first
                cleaned_response = response.strip().strip('```yaml',").strip('```',").strip()
                extracted_data = yaml.safe_load(cleaned_response)
                
                if not isinstance(extracted_data, dict):
                    logger.error(f"Invalid YAML structure received: {cleaned_response}")
                    print("\nError: Received invalid data format from API")
                    return
                
                # Continue with patient creation/update
                if self.patient_id:
                    # Update existing patient
                    update_success = self.db_manager.update_patient(
                        self.patient_id, 
                        extracted_data
                    )
                    if update_success:
                        logger.info(f"Successfully updated patient {self.patient_id}")
                        print(f"\nSuccessfully updated patient {self.patient_id}")
                    else:
                        logger.warning(f"No updates made for patient {self.patient_id}")
                        print("\nNo updates were needed for the patient record")
                else:
                    # Create new patient
                    new_patient_id = self._create_patient_from_yaml(extracted_data)
                    if new_patient_id:
                        print(f"\nCreated new patient with ID: {new_patient_id}")
                    else:
                        print("\nFailed to create new patient - check logs for details")
                
            except yaml.YAMLError as e:
                logger.error(f"YAML parsing error: {str(e)}")
                print(f"\nError processing extracted data: Invalid YAML format")
                return
            except Exception as e:
                logger.error(f"Error processing YAML data: {str(e)}")
                print(f"\nError processing extracted data: {str(e)}")
                return
            
            # Save results
            timestamp = self._get_timestamp()
            raw_path, processed_path = self._save_yaml(response, timestamp)
            
            print("\nExtracted Information:")
            print(response)
            print(f"\nFiles saved:")
            print(f"- Transcript: data/transcripts/transcript_{timestamp}.txt")
            print(f"- Raw YAML: {raw_path}")
            print(f"- Processed YAML: {processed_path}")
            
        except Exception as e:
            logger.error(f"Error in conversation: {str(e)}")
            print(f"\nError occurred: {str(e)}")

class APIHandler:
    def handle_request(self, request):  # Added self parameter
        return self.process_request()

def main():
    """Main entry point of the application."""
    try:
        import argparse
        parser = argparse.ArgumentParser(description='Medical Assistant Transcription System')
        parser.add_argument('--patient-id', type=int, help='Patient ID for database updates')
        args = parser.parse_args()
        
        assistant = SpeechAssistant()
        assistant.run(patient_id=args.patient_id)
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}")
        print("An error occurred. Please check the logs for details.")

if __name__ == "__main__":
    main()
