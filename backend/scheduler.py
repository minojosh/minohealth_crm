"""
Moremi Scheduling Assistant - Production Version
A speech-based medical assistant that uses Azure Cognitive Services for speech-to-text
and text-to-speech functionality, integrated with the Moremi AI model to schedule appointments.
"""

from datetime import datetime
import requests
import os
import time
from dataclasses import dataclass, asdict
from .database import Doctor, Session, Patient, Appointment
from .appointment_manager import SchedulerManager
from .moremi import ConversationManager
import json
from typing import List, Optional
import logging
import argparse
import traceback
from .STT_client import SpeechRecognitionClient
from .XTTS_adapter import TTSClient
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from pathlib import Path

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

# Load environment variables
load_dotenv()

# Initialize API and TTS clients
client = SpeechRecognitionClient()
speech_client = TTSClient(os.getenv("TTS_SERVER_URL"))
db_manager = SchedulerManager(3) # Default value of days ahead is 3
doctors = session.query(Doctor).all()
patients = session.query(Patient).all()

# Initialize ConversationManager with proper API credentials
moremi_base_url = os.getenv("MOREMI_API_BASE_URL")
moremi_api_key = os.getenv("MOREMI_API_KEY") or os.getenv("OPENAI_API_KEY")

try:
    # Initialize with both URL and key specified
    LLM = ConversationManager(moremi_base_url, moremi_api_key)
    logger.info(f"Successfully initialized ConversationManager with URL: {moremi_base_url}")
except Exception as e:
    # Log the error but don't crash during module import
    logger.error(f"Failed to initialize ConversationManager: {str(e)}")
    logger.error(traceback.format_exc())
    # Set LLM to None - individual functions will need to check and reinitialize if needed
    LLM = None

# Initialize scheduler
db_manager = SchedulerManager(3)  # Default value of days ahead is 3

# Get all current doctors into a list
docs = []
for doctor in doctors:
    details={
            'doctor_id': doctor.doctor_id,  # Add doctor_id to details
            'doctor_name': doctor.name,
            'specialty': doctor.specialization
            }
    docs.append(details)
    
    
def schedule_appointment(patient_id: int, appointment_details: dict) -> dict:
    """
    Schedule the appointment for a given patient.
    
    Args:
        patient_id: The ID of the patient
        appointment_details: A dictionary of the appointment details
        
        
    Returns:
        dict: A dictionary containing the status and message of the operation
    """
    
    try:
        # Check if patient exists
        patient = session.query(Patient).filter_by(patient_id=patient_id).first()
        if not patient:
            return {
                "success": False,
                "message": f"Patient with name and ID {patient.name}(ID:{patient_id}) not found in database"
            }
        elif patient.name != appointment_details['patientName']:
            logger.info(f"Updating patient name as {patient.name} instead of {appointment_details['patientName']} in database")
        
        # Find an existing appointment for the patient
        appointment = session.query(Appointment).filter_by(patient_id=patient_id).first()
        if appointment:
            logger.info(f"An appointment exists for the patient on {appointment.datetime} with {[doc['doctor_name'] for doc in docs if doc['doctor_id'] == appointment.doctor_id]}")
        
        try:
            logger.info("Attempting to update the database")
            # Try to extract doctor_id
            matching_doctors = [doc["doctor_id"] for doc in docs if doc['doctor_name'] == appointment_details['doctorName']]
            
            if not matching_doctors:
                logger.error(f"No matching doctor found for name: {appointment_details['doctorName']}")
                raise ValueError(f"No doctor found with name {appointment_details['doctorName']}")
            
            doctor_id = matching_doctors[0]
            
            # Convert datetime
            try:
                appointment_datetime = datetime.strptime(str(appointment_details['appointmentDateTime']), '%Y-%m-%d %H:%M:%S').replace(microsecond=0)
            except ValueError as dt_error:
                logger.error(f"Failed to parse datetime: {appointment_details['appointmentDateTime']}")
                logger.error(f"Datetime parsing error: {dt_error}")
                raise
            
            new_appointment = Appointment(
                patient_id=patient_id,
                doctor_id=doctor_id,
                datetime=appointment_datetime,
                appointment_type=appointment_details["appointmentType"],
                notes=appointment_details["summary"]
            )
            session.add(new_appointment)
            session.commit()
            logger.info("Successfully updated the database")
            print("\nSuccessfully updated the database")

        except Exception as e:
            session.rollback()
            logger.error("Failed to update the database")
            logger.error(f"Error details: {str(e)}")
            logger.error(traceback.format_exc())
            raise  # Re-raise the exception for higher-level error handling
        
        return {
            "success": True,
            "message": "Appointment updated successfully"
        }
        
    except Exception as e:
        session.rollback()
        return {
            "success": False,
            "message": f"Database error: {str(e)}"
        }
    finally:
        session.close()


@dataclass
class Message:
    user_input: str
    response: Optional[str] = None

    def add_response(self, response: str):
        self.response = response

    def to_dict(self):
        return asdict(self)

class SpeechAssistant:
    """Main speech assistant class that handles speech recognition and synthesis."""
    def __init__(self):
        """Initialize the speech assistant with necessary configurations."""
        current_dir = Path(__file__).resolve().parent
        env_path = current_dir / '.env'
        
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading .env file from: {env_path}")
        
        self.url = os.getenv("LLM_URL")
        self.key = os.getenv("API_KEY")
        self.LLM = ConversationManager(self.url)
        
        if not self.url:
            logger.error(f"API_URL is not set. Checked .env file at: {env_path}")
            raise ValueError("API_URL is not set")
            
        self._load_prompts()
        self.LLM.custom_params["system_prompt"] = self.system_prompt
        
        # Initialize other attributes
        self.suit_doc = []
        self.patient_id = None
        self.SILENCE_TIMEOUT = 3  # seconds
        self.messages: List[Message] = []

    def _load_prompts(self) -> None:
        """Load system prompts from configuration file."""
        try:
            with open('backend/prompt.json', 'r') as file:
                schema = json.load(file)
            self.system_prompt = schema.get("systemprompt", "")
            self.summary_system_prompt = schema.get("summarysystemprompt", "")
        except FileNotFoundError:
            logger.warning("prompt.json not found. Using default empty prompts.")
            self.system_prompt = ""
            self.summary_system_prompt = ""

    def get_moremi_response(self, messages: List[Message],  system_prompt: Optional[str] = None) -> str:
        """Get response from Moremi API with retry logic."""

        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        
        query = [msg.to_dict() for msg in messages]
        
        data = {
            "query": query,
            "temperature": 0.5,
            "max_new_token": 100,
            "top_p": 0.9,
            "history": True
        }
        if system_prompt:
            data["systemPrompt"] = system_prompt

        headers ={
        "Authorization": f"Bearer {self.key}",
        "azureml-model-deployment": "llava-deployment",
        "Content-Type": "application/json"     
        }
        
        # Log request details (safely hiding the full authorization token)
        auth_header = headers["Authorization"]
        safe_auth = auth_header[:15] + "..." if len(auth_header) > 15 else auth_header
        safe_headers = {**headers, "Authorization": safe_auth}
        
        logger.info(f"Request headers: {safe_headers}")
        logger.info(f"Request data size: {len(str(data))} characters")
        logger.info(f"Request contains {len(query)} messages")
        
        # Log a truncated version of the data for debugging
        if query:
            last_msg = query[-1]
            logger.info(f"Last message in query: {json.dumps(last_msg, indent=2)[:200]}...")
        
        if system_prompt:
            logger.info(f"Using system prompt (first 100 chars): {system_prompt[:100]}...")
            
        try:
            logger.info("Making API request to Moremi...")
            start_time = time.time()
            response = requests.post(self.url, json=data)
            elapsed_time = time.time() - start_time
            logger.info(f"API request completed in {elapsed_time:.2f} seconds")
            
            # Log response details
            logger.info(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.reason}")
                logger.error(f"Response content: {response.text[:200]}...")
                response.raise_for_status()
                
            response_text = response.text
            logger.info(f"Response length: {len(response_text)} chars")
            logger.info(f"Response preview (first 200 chars): {response_text[:200]}...")
            
            return response_text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Moremi API: {str(e)}")
            logger.error(f"Request that caused error: URL={self.url}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def save_conversation(self, messages) -> None:
        """Save the conversation history to a file."""
        try:
            # Convert messages to serializable format
            conversation_to_save = []
            for msg in messages:
                if hasattr(msg, 'to_dict'):
                    conversation_to_save.append(msg.to_dict())
                else:
                    conversation_to_save.append(msg)
                    
            # Save to file
            with open('conversation.txt', 'w') as f:
                json.dump(conversation_to_save, f, indent=4)
            logger.info("Conversation saved successfully")
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")

    def manage_context(self, context: Optional[str] = None):
        if context is not None:
            context = context
        else:
            with open('scheduler/contexts.json', 'r') as f:
                contexts = json.load(f)
            context = contexts['context3']
        
        # Extract doctor names and specialties
        names = [doc['doctor_name'] for doc in docs]
        specialties = [doc['specialty'] for doc in docs]

        # Initial Doctor paitient conversation
        question = f'''
Task: You are a doctor's medical assistant that strictly follows the given instructions. Never deviate or add anything extra. Follow these rules exactly as written. Think step by step.
The current user is not the patient but is scheduling an appointment for the patient.  NEVER address the patient directly instead all question should be routed to through the user.
Your task is to aid the user schedule an appointment by following these steps:
1. Select the doctor 
   - Understand the conversation below for a scheduling task.
   - If a doctor's name is mentioned, check this list of doctors, <{names}>, and only output the doctor's full name nothing before or after.  
   - If a doctor's name is not explicitly mentioned, do this subsection:
      -- From the conversation try and match the described specialty with each specialty in our list of available specialties, <{specialties}>.
      -- If there isn't a match then extract the exact name of the specialty from the conversation and ONLY output, search_specialty: <specialty>, nothing before or after.
      -- If there is a match then refer to the specialty exactly as it is in our list and ONLY output, search_specialty: <specialty>, nothing before or after.
   - Only do this after the user has confirmed the patients appointment date. Extract the patients name and confirm with the user, if the name is not known ask the user directly. Example: What is the name of the patient.   
     
Conversation:
{context}  
'''
        print(context)
        return question
            
    def run(self) -> None:
        """Run the main conversation loop."""
        # Initial Greeting 
        print("Welcome to Moremi AI Scheduler!, I am listening")
        speech_client.TTS("Welcome to Moremi AI Scheduler!, I am listening")
        logger.info("Starting conversation")
        print("Starting conversation. Press Ctrl+C to exit.")
        
        # Add initial context and get response
        self.LLM.custom_params["system_prompt"] = self.system_prompt
        self.LLM.add_user_message(text=self.manage_context())
        response = self.LLM.get_assistant_response()
        logger.info(f"Moremi response: {response}")

        # Process initial response for specialty/doctor search
        self.process_initial_response(response)
        
        # Process slots if doctors were found
        if self.suit_doc:
            response = ''
            for doc in self.suit_doc:
                while True:
                    slots = db_manager._get_new_date(doc['doctor_id'])
                    if slots:
                        response += f'\n{db_manager.format_available_slots(slots, doc["doctor_name"])}\n'
                        break
                    elif db_manager.days_ahead == 90:
                        response += f'\n{db_manager.format_available_slots(slots, doc["doctor_name"])}\n'
                        break
                    else:
                        db_manager.days_ahead += 5

            # Update LLM conversation
            self.LLM.add_user_message(text="What slots are available")
            self.LLM.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            logger.info(response)
            speech_client.TTS(response)
            print("\nReady for next input...")
        else:
            # No doctors found - add message to LLM
            self.LLM.add_user_message(text="Find a suitable doctor")
            response = "Unfortunately there are no doctors available at the moment. Please speak to the present doctor for clarification of subsequent steps."
            self.LLM.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            speech_client.TTS(response)
            print("\nReady for next input...")
        
        # Main conversation loop
        while True:
            try:
                # Get speech input
                client = SpeechRecognitionClient()
                client.recognize_speech()
                user_input = client.get_transcription()
                if not user_input.strip():
                    logger.warning("Empty input received. Please say something.")
                    continue

                logger.info(f"User input: {user_input}")
                
                # Get Moremi response
                self.LLM.add_user_message(text=user_input)
                response = self.LLM.get_assistant_response()
                logger.info(f"Moremi response: {response}")

                # Synthesize response
                speech_client.TTS(response)

            except KeyboardInterrupt:
                logger.info("Conversation ended by user")
                print('\nEnding conversation...')
                
                # Get conversation summary
                try:
                    # Change to summary system prompt
                    self.LLM.custom_params["system_prompt"] = self.summary_system_prompt
                    self.LLM.add_user_message(
                        text="Extract the agreed upon schedule from the interaction history. If there was no agreed upon appointment respond with the word null."
                    )
                    summary = self.LLM.get_assistant_response()
                    
                    print(summary)
                    if summary == "null":
                        logger.info("No schedule was determined")
                        speech_client.TTS('No schedule was determined')
                    else:
                        try:
                            parsed = json.loads(summary)
                            # Convert string to dict if needed
                            if isinstance(parsed, str):
                                parsed_dict = json.loads(parsed)
                            else:
                                parsed_dict = parsed
                                
                            # Write to temporary file
                            with open('schedules_t.json', 'w') as f:
                                json.dump(parsed_dict, f, indent=4)
                                
                            # Synthesize success message
                            print(f"The appointment for {parsed_dict['patientName']} with {parsed_dict['doctorName']} on {parsed_dict['appointmentDateTime']} has been booked successfully")
                            
                            # Update the database
                            schedule_appointment(self.patient_id, parsed_dict)
                            
                        except json.JSONDecodeError as e:
                            print("Invalid JSON format:", e)
                    
                    print(f'\nSummary: {summary}')
                    
                except Exception as e:
                    logger.error(f"Failed to generate summary: {str(e)}")
                
                self.save_conversation(self.LLM.conversation_history)
                break
                
            except Exception as e:
                logger.error(f"Error in conversation loop: {str(e)}")
                error_response = 'Sorry. The system is currently unavailable.'
                try:
                    speech_client.TTS(error_response)
                except:
                    pass
                break

    def process_initial_response(self, response):
        if 'search_specialty' in str(response).lower():
            self.handle_specialty_search(response)
        else:
            self.handle_doctor_search(response)

    def handle_specialty_search(self, response):
        text = "Finding a suitable doctor"
        logger.info(text)
        speech_client.TTS(text)
        
        for doc in docs:
            specialty = doc['specialty']
            if specialty.lower() in response.lower():
                self.suit_doc.append(doc)
        
        if not self.suit_doc:
            self.handle_no_doctors_found("specialty")

    def handle_doctor_search(self, response):
        logger.info('Finding available days for the doctor')
        speech_client.TTS('Finding available days for the doctor')
        
        for doc in docs:
            if doc['doctor_name'] in response:
                self.suit_doc.append(doc)
        
        if not self.suit_doc:
            self.handle_no_doctors_found("doctor")


    def handle_no_doctors_found(self, search_type):
        logger.info(f"The required {search_type} is not available in our facility. Please speak to the present doctor about a referral.")
        speech_client.TTS(f"The required {search_type} is not available in our facility. Please speak to the present doctor about a referral.")


def main():
    """Main entry point of the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the appointment scheduler.')
    parser.add_argument('--patient-id', type=int, required=True,
                    help='Patient identifier')
    
    args = parser.parse_args()
    try: 
        assistant = SpeechAssistant()
        assistant.patient_id = args.patient_id
        assistant.run()
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}")
        print("An error occurred. Please check the logs for details.")
   
   
if __name__ == '__main__':
    main()
 



