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
# import select
# import termios
# import tty
import sys, asyncio
if sys.platform.startswith("linux"):
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

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
from .database_utils import PatientDatabaseManager
from .XTTS_adapter import TTSClient
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

class Assistant:
    """Main speech assistant class that handles speech recognition."""
    
    def __init__(self):
        """Initialize the speech assistant with necessary configurations."""
        self._setup_directories()
        self._setup_logging()
        
        # Update config path to use project root
        config_path = Path(__file__).parent.parent / '.env'
        if not config_path.exists():
            logger.warning(f"Environment file not found at {config_path}, using environment variables")
        load_dotenv(dotenv_path=config_path, encoding='utf-8')
        
        # Initialize STT client settings
        self.stt_server_url = os.getenv("STT_URL", "")
        
        # Initialize other components
        self.messages: List[Message] = []
        self.api = os.getenv("MOREMI_API_BASE_URL")  # Fix: Use correct env var name
        self._load_prompts()
        self.db_manager = PatientDatabaseManager()
        self.patient_id = None

    def _setup_logging(self):
        """Configure logging to file and console."""
        log_dir = Path(__file__).parent / 'data' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_file = log_dir / f"app_{self._get_timestamp()}.log"
        
        logger.info(f"Setting up logging to file: {log_file}")
        
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
        # Create directories relative to the current file (backend folder)
        data_dir = Path(__file__).parent / 'data'
        dirs = [
            'audio',
            'transcripts',
            'responses',
            'yaml_output/raw',
            'yaml_output/processed',
            'logs'
        ]
        
        # Ensure the main data directory exists
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensuring data directory exists at: {data_dir}")
        
        # Create all subdirectories
        for dir_name in dirs:
            dir_path = data_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")

    def _get_timestamp(self) -> str:
        """Generate a formatted timestamp."""
        return datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

    def recognize_speech(self) -> str:
        """Record and transcribe speech from microphone using STT client."""
        timestamp = self._get_timestamp()
        audio_path = Path(__file__).parent / 'data' / 'audio' / f'recording_{timestamp}.wav'
        transcript_path = Path(__file__).parent / 'data' / 'transcripts' / f'transcript_{timestamp}.txt'
        
        logger.info("Starting speech recognition using STT client")
        print("\nListening... Press 'f' to finish recording")
        
        try:
            # Initialize the STT client
            stt_client = SpeechRecognitionClient(server_url=self.stt_server_url)
            
            # Start speech recognition
            logger.info("Starting speech recognition")
            transcript = stt_client.recognize_speech()
            
            if transcript:
                # Ensure the transcripts directory exists
                transcript_path.parent.mkdir(parents=True, exist_ok=True)
                
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
        """Get response from Moremi API using the ConversationManager."""
        if not self.api:
            raise ValueError("Moremi API URL not configured")

        try:
            # Initialize the ConversationManager with the API URL
            conversation_manager = ConversationManager(base_url=self.api)
            
            # Set the system prompt if provided
            if system_prompt:
                conversation_manager.custom_params["system_prompt"] = system_prompt
            
            # Get the current message to process
            current_message = messages[-1] if messages else None
            if not current_message:
                raise ValueError("No message to process")
            
            # Format the query with strict YAML requirements and better extraction guidance
            extraction_prompt = (
                f"Extract ALL medical information from this transcript and format it STRICTLY as YAML.\n\n"
                f"Transcript: {current_message.user_input}\n\n"
                f"Rules:\n"
                f"1. ONLY output valid YAML\n"
                f"2. DO NOT include any explanatory text or comments\n"
                f"3. Use null for missing values, not None\n"
                f"4. If no medical information is found, still output the structure with null values\n"
                f"5. IMPORTANT: Extract ANY information that could be relevant, even if incomplete\n"
                f"6. For age, convert to a birthdate estimate if exact DOB is not provided\n"
                f"7. Format must be exactly:\n\n"
                f"name: [string or null]\n"
                f"dob: [YYYY-MM-DD or null]\n"
                f"address: [string or null]\n"
                f"phone: [string or null]\n"
                f"email: [string or null]\n"
                f"insurance: [string or null]\n"
                f"condition: [string or null]\n"
                f"symptoms: [list of strings or null]\n"
                f"reason_for_visit: [string or null]\n"
                f"appointment_details:\n"
                f"  type: [string or null]\n"
                f"  time: [HH:MM or null]\n"
                f"  doctor: [string or null]\n"
                f"  scheduled_date: [YYYY-MM-DD or null]\n"
            )
            
            logger.info(f"Sending transcript to Moremi API: {current_message.user_input[:100]}...")
            
            # Add the user message with extraction prompt
            conversation_manager.add_user_message(text=extraction_prompt)
            
            # Get the response (with non-streaming to get the full response at once)
            response = conversation_manager.get_assistant_response(
                stream=False,
                max_tokens=500,
                temperature=0.2
            )
            
            # Clean up the response - remove any non-YAML content
            response_lines = response.split('\n')
            yaml_lines = []
            in_yaml = False
            
            for line in response_lines:
                # Start capturing at the first YAML key
                if line.strip().startswith('name:'):
                    in_yaml = True
                
                if in_yaml:
                    # Stop if we hit an empty line after YAML content
                    if not line.strip() and yaml_lines:
                        break
                    if line.strip():  # Only add non-empty lines
                        yaml_lines.append(line)
            
            yaml_content = '\n'.join(yaml_lines)
            logger.debug(f"Extracted YAML content:\n{yaml_content}")
            
            # Validate YAML
            import yaml
            try:
                parsed_yaml = yaml.safe_load(yaml_content)
                if not isinstance(parsed_yaml, dict):
                    raise ValueError(f"Invalid YAML structure")
                
                # Ensure all required fields exist with proper types
                required_fields = {
                    'name': str, 
                    'dob': str,
                    'address': str,
                    'phone': str,
                    'email': str,
                    'insurance': str,
                    'condition': str,
                    'symptoms': list,
                    'reason_for_visit': str,
                    'appointment_details': dict
                }
                
                # Initialize with null values if missing
                cleaned_data = {}
                for field, field_type in required_fields.items():
                    value = parsed_yaml.get(field)
                    if value is None or value == "None" or value == "null":
                        cleaned_data[field] = None
                    elif isinstance(value, field_type):
                        cleaned_data[field] = value
                    else:
                        cleaned_data[field] = None
                
                # Special handling for age to DOB conversion
                if not cleaned_data.get('dob') and 'age' in str(current_message.user_input).lower():
                    # Try to extract age and convert to estimated DOB
                    import re
                    from datetime import datetime, timedelta
                    
                    age_match = re.search(r'age\s+(\d+)', str(current_message.user_input).lower())
                    if age_match:
                        age = int(age_match.group(1))
                        # Calculate approximate birth year
                        birth_year = datetime.now().year - age
                        cleaned_data['dob'] = f"{birth_year}-01-01"  # Use January 1st as default
                        logger.info(f"Converted age {age} to estimated DOB: {cleaned_data['dob']}")
                
                # Handle appointment details specifically
                appt_details = cleaned_data.get('appointment_details', {})
                if not isinstance(appt_details, dict):
                    appt_details = {}
                
                cleaned_data['appointment_details'] = {
                    'type': appt_details.get('type'),
                    'time': appt_details.get('time'),
                    'doctor': appt_details.get('doctor'),
                    'scheduled_date': appt_details.get('scheduled_date')
                }
                
                # Convert back to YAML string
                return yaml.dump(cleaned_data, default_flow_style=False)
                
            except yaml.YAMLError as e:
                logger.error(f"Invalid YAML received: {str(e)}")
                logger.error(f"Raw content: {yaml_content}")
                raise ValueError(f"Invalid YAML received from API: {str(e)}")
            
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
            prompt_path = Path(__file__).parent / 'data' / 'prompts' / 'prompt_config.json'
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
        # Use the backend/data directory for saving files
        base_dir = Path(__file__).parent / 'data' / 'yaml_output'
        
        # Ensure directories exist
        (base_dir / 'raw').mkdir(parents=True, exist_ok=True)
        (base_dir / 'processed').mkdir(parents=True, exist_ok=True)
        
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
            
            # Extract patient data, converting None to None explicitly
            patient_data = {
                'name': yaml_data.get('name') if yaml_data.get('name') != 'null' else None,
                'phone': yaml_data.get('phone') if yaml_data.get('phone') != 'null' else None,
                'email': yaml_data.get('email') if yaml_data.get('email') != 'null' else None,
                'address': yaml_data.get('address') if yaml_data.get('address') != 'null' else None,
                'date_of_birth': yaml_data.get('dob') if yaml_data.get('dob') != 'null' else None
            }
            
            # Handle required fields - provide defaults if missing
            if not patient_data.get('name'):
                logger.warning("Missing required field: name")
                return None
                
            # Add placeholder for required phone if missing
            if not patient_data.get('phone'):
                patient_data['phone'] = "Unknown"
                logger.warning("Missing phone number, using placeholder")
            
            # Remove None values and empty strings
            patient_data = {k: v for k, v in patient_data.items() if v not in (None, '', 'null')}
            
            # Check if we have any valid data to create a patient
            if not any(patient_data.values()):
                logger.warning("No valid patient data extracted from transcript")
                return None
                
            logger.debug(f"Attempting to create patient with data: {patient_data}")
            
            # Create patient
            patient_id = self.db_manager.create_patient(patient_data)
            
            if patient_id:
                logger.info(f"Created patient with ID: {patient_id}")
                
                # Handle appointment if present
                appt = yaml_data.get('appointment_details', {})
                if appt and isinstance(appt, dict) and any(v not in (None, '', 'null') for v in appt.values()):
                    logger.debug(f"Creating appointment with details: {appt}")
                    appt_id = self.db_manager.create_appointment(
                        patient_id=patient_id,
                        appointment_type=appt.get('type') if appt.get('type') != 'null' else None,
                        scheduled_date=appt.get('scheduled_date') if appt.get('scheduled_date') != 'null' else None,
                        scheduled_time=appt.get('time') if appt.get('time') != 'null' else None,
                        doctor_name=appt.get('doctor') if appt.get('doctor') != 'null' else None
                    )
                    if appt_id:
                        logger.info(f"Created appointment {appt_id} for patient {patient_id}")
                
                # Handle medical condition and symptoms if present
                condition = yaml_data.get('condition')
                symptoms = yaml_data.get('symptoms', [])
                if condition not in (None, '', 'null') or (symptoms and any(s not in (None, '', 'null') for s in symptoms)):
                    logger.debug(f"Adding medical condition: {condition} and symptoms: {symptoms}")
                    # Add medical conditions
                    self.db_manager.add_medical_condition(
                        patient_id=patient_id,
                        condition=condition if condition != 'null' else None,
                        symptoms=[s for s in symptoms if s not in (None, '', 'null')]
                    )
            
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
                # Clean up the response first - fixed the string manipulation
                cleaned_response = response
                # Remove any markdown code blocks if present
                if "```yaml" in cleaned_response:
                    cleaned_response = cleaned_response.replace("```yaml", "").replace("```", "")
                
                # Ensure the response is properly stripped
                cleaned_response = cleaned_response.strip()
                
                logger.debug(f"Cleaned YAML response: {cleaned_response}")
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
            print(f"- Transcript: backend/data/transcripts/transcript_{timestamp}.txt")
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
        
        assistant = Assistant()
        assistant.run(patient_id=args.patient_id)
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}")
        print("An error occurred. Please check the logs for details.")

if __name__ == "__main__":
    main()
