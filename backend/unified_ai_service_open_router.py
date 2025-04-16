"""
Unified AI Service - Integration Layer for minoHealth AI services
This module provides a unified interface to various AI services including:
- OpenRouter AI conversational services
- Speech-to-Text (STT) services 
- Text-to-Speech (TTS) services

The design allows for different use cases (scheduler, differential diagnosis, etc.)
to be handled through a common interface with specialized behavior.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import base64
from dotenv import load_dotenv

# Import existing implementations
from .XTTS_adapter import TTSClient
from .STT_client import SpeechRecognitionClient
from .moremi_open_router import ConversationManager as OpenRouterConversationManager
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration for AI services including system prompts."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config manager with path to prompt configuration file.
        
        Args:
            config_path: Path to the configuration file (defaults to prompt.json in same directory)
        """
        # Load environment variables
        load_dotenv()
        
        # Set default config path if not provided
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'prompt.json')
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                logger.info(f"Successfully loaded configuration from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Configuration file not found at {config_path}. Using empty configuration.")
            self.config = {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file {config_path}. Using empty configuration.")
            self.config = {}
    
    def get_prompt(self, prompt_key: str) -> str:
        """
        Get system prompt by key.
        
        Args:
            prompt_key: Key of the prompt in the configuration
            
        Returns:
            The system prompt string or empty string if not found
        """
        return self.config.get(prompt_key, "")
    
    def get_tts_url(self) -> Optional[str]:
        """Get TTS service URL from environment variables."""
        return os.getenv("SPEECH_SERVICE_URL") or os.getenv("TTS_SERVER_URL")
    
    def get_stt_url(self) -> Optional[str]:
        """Get STT service URL from environment variables."""
        return os.getenv("SPEECH_SERVICE_URL")
    
    def get_openrouter_api_key(self) -> Optional[str]:
        """Get OpenRouter API key from environment variables."""
        return os.getenv("OPENROUTER_API_KEY")
    
    def get_model_name(self) -> str:
        """Get the default model name to use."""
        return os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-lite-001")


class AIService:
    """Base class for AI services providing unified interface to OpenRouter, TTS and STT."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize AI service with configuration.
        
        Args:
            config_manager: Configuration manager instance or None to create a new one
        """
        self.config_manager = config_manager or ConfigManager()
        self.tts_client = self._init_tts_client()
        self.stt_client = self._init_stt_client()
        self.llm_client = self._init_llm_client()
    
    def _init_tts_client(self) -> TTSClient:
        """Initialize TTS client."""
        api_url = self.config_manager.get_tts_url()
        logger.info(f"Initializing TTS client with URL: {api_url}")
        return TTSClient(api_url=api_url)
    
    def _init_stt_client(self) -> SpeechRecognitionClient:
        """Initialize STT client."""
        server_url = self.config_manager.get_stt_url()
        logger.info(f"Initializing STT client with URL: {server_url}")
        return SpeechRecognitionClient(server_url=server_url)
    
    def _init_llm_client(self) -> OpenRouterConversationManager:
        """Initialize OpenRouter client."""
        api_key = self.config_manager.get_openrouter_api_key()
        model = self.config_manager.get_model_name()
        
        if not api_key:
            logger.warning("No OpenRouter API key found. Using default.")
        
        logger.info(f"Initializing OpenRouter client with model: {model}")
        client = OpenRouterConversationManager(api_key=api_key)
        client.set_model(model)
        return client
    
    def process_with_prompt(self, prompt_key: str, user_input: str, 
                           max_tokens: int = 300, temperature: float = 1.0,
                           should_speak: bool = False) -> str:
        """
        Process user input with the specified system prompt.
        
        Args:
            prompt_key: Key of the system prompt to use
            user_input: User input text to process
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
            should_speak: Whether to speak the response using TTS
            
        Returns:
            The response from OpenRouter
        """
        # Get the appropriate system prompt
        system_prompt = self.config_manager.get_prompt(prompt_key)
        logger.info(f"Using prompt key '{prompt_key}' for processing")
        
        # Set the system prompt
        if system_prompt:
            self.llm_client.set_system_prompt(system_prompt)
        
        # Process the input and get response
        self.llm_client.add_user_message(text=user_input)
        response = self.llm_client.get_assistant_response(
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            should_speak=should_speak
        )
        
        return response
    
    def recognize_speech(self) -> str:
        """
        Recognize speech using the STT client.
        
        Returns:
            Transcribed text
        """
        logger.info("Starting speech recognition")
        return self.stt_client.recognize_speech()
    
    def speak_text(self, text: str, play_locally: bool = True) -> Optional[Dict[str, Any]]:
        """
        Speak text using the TTS client.
        
        Args:
            text: Text to speak
            play_locally: Whether to play audio locally
            
        Returns:
            Audio data dictionary if successful, None if failed
        """
        logger.info(f"Speaking text: {text[:30]}...")
        try:
            if play_locally:
                self.tts_client.stream_text(text)
                self.tts_client.wait_for_completion()
                return None
            else:
                return self.tts_client.get_audio_for_frontend(text)
        except Exception as e:
            logger.error(f"Error speaking text: {e}")
            return None


class SchedulerService(AIService):
    """Specialized service for handling appointment scheduling."""
    
    def schedule_appointment(self, conversation_text: str, should_speak: bool = False) -> str:
        """
        Process appointment scheduling conversation.
        
        Args:
            conversation_text: Text of the conversation
            should_speak: Whether to speak the response
            
        Returns:
            Processed response
        """
        return self.process_with_prompt("systemprompt", conversation_text, should_speak=should_speak)
    
    def summarize_appointment(self, conversation_text: str) -> str:
        """
        Summarize an appointment scheduling conversation.
        
        Args:
            conversation_text: Text of the conversation
            
        Returns:
            Appointment summary in JSON format
        """
        return self.process_with_prompt("summarysystemprompt", conversation_text)
    
    def handle_appointment_reminder(self, patient_info: Dict[str, str]) -> str:
        """
        Generate appointment reminder response.
        
        Args:
            patient_info: Dictionary with patient information
            
        Returns:
            Response text
        """
        template = self.config_manager.get_prompt("initial_appointment_reminder")
        reminder_text = template.format(**patient_info)
        return self.process_with_prompt("appointment_prompt", reminder_text)


class DifferentialDiagnosisService(AIService):
    """Specialized service for handling differential diagnosis conversations."""
    
    def diagnose(self, patient_symptoms: str, should_speak: bool = False) -> str:
        """
        Process differential diagnosis based on patient symptoms.
        
        Args:
            patient_symptoms: Description of patient symptoms
            should_speak: Whether to speak the response
            
        Returns:
            Diagnosis response
        """
        return self.process_with_prompt("differential_system_prompt", 
                                       patient_symptoms, 
                                       should_speak=should_speak)
    
    def summarize_diagnosis(self, diagnosis_conversation: str) -> str:
        """
        Summarize a diagnosis conversation.
        
        Args:
            diagnosis_conversation: Text of the diagnosis conversation
            
        Returns:
            Diagnosis summary
        """
        return self.process_with_prompt("differential_summary_systemprompt", 
                                       diagnosis_conversation)
    
    def generate_soap_note(self, conversation_text: str) -> Dict[str, Any]:
        """
        Generate a SOAP medical note from conversation.
        
        Args:
            conversation_text: Text of the medical conversation
            
        Returns:
            SOAP note in dictionary format
        """
        soap_prompt = self.config_manager.get_prompt("soap_template_prompt")
        soap_template = self.config_manager.get_prompt("soap_template")
        
        # Combine prompt with template and conversation
        full_prompt = f"{soap_prompt}\n\nConversation:\n{conversation_text}\n\nSOAP_TEMPLATE: {json.dumps(soap_template)}"
        
        response = self.process_with_prompt("soap_template_prompt", full_prompt)
        
        try:
            # Try to parse the response as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse SOAP note response as JSON: {response}")
            return {"error": "Failed to generate valid SOAP note", "raw_response": response}


class MedicationReminderService(AIService):
    """Specialized service for handling medication reminders."""
    
    def handle_medication_reminder(self, patient_info: Dict[str, str], should_speak: bool = False) -> str:
        """
        Generate medication reminder response.
        
        Args:
            patient_info: Dictionary with patient information
            should_speak: Whether to speak the response
            
        Returns:
            Response text
        """
        template = self.config_manager.get_prompt("medication_reminder_prompt")
        reminder_text = template.format(**patient_info)
        return self.process_with_prompt("medication_reminder_prompt", reminder_text, should_speak=should_speak)

class MedicalAssistantService(AIService):
    """Specialized service for handling medical assistant tasks including data extraction and SOAP notes."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize the medical assistant service with database manager."""
        super().__init__(config_manager)
        # Import here to avoid circular imports
        from .database_utils import PatientDatabaseManager
        self.db_manager = PatientDatabaseManager()
    
    def extract_data(self, transcript: str) -> Dict[str, Any]:
        """
        Extract structured medical data from transcript in YAML format.
        
        Args:
            transcript: The transcript text to extract data from
            
        Returns:
            Dictionary with extracted medical data
        """
        # Format the extraction prompt for YAML output
        extraction_prompt = (
            f"Extract ALL medical information from this transcript and format it STRICTLY as YAML.\n\n"
            f"Transcript: {transcript}\n\n"
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
        
        # Send the request to OpenRouter for processing
        logger.info(f"Extracting data from transcript: {transcript[:100]}...")
        
        # Configure extraction-specific parameters
        self.llm_client.set_system_prompt("You are an expert medical data extraction system.")
        
        # Add the user message with extraction prompt
        self.llm_client.add_user_message(text=extraction_prompt)
        
        # Get the response (with non-streaming to get the full response at once)
        response = self.llm_client.get_assistant_response(
            stream=False,
            max_tokens=500,
            temperature=0.2
        )

        # --- START YAML CLEANUP ---
        import re
        import yaml

        logger.debug(f"Raw LLM response for YAML extraction:\n{response}") # Log raw response

        # Try a slightly broader regex first
        match = re.search(r"```(.*?)```", response, re.DOTALL)

        yaml_content_to_parse = ""
        if match:
            yaml_content_to_parse = match.group(1).strip()
            # Remove potential leading 'yaml' identifier if captured by broader regex
            if yaml_content_to_parse.lower().startswith("yaml"):
                 yaml_content_to_parse = yaml_content_to_parse[4:].strip()
            logger.info("Extracted YAML content from within markdown fences using regex.")
        else:
            # Regex failed, try manual string stripping as a fallback
            logger.warning("Regex did not find fences. Trying manual stripping.")
            stripped_response = response.strip()
            if stripped_response.startswith("```yaml") and stripped_response.endswith("```"):
                 yaml_content_to_parse = stripped_response[7:-3].strip()
                 logger.info("Extracted YAML content by manually stripping ```yaml.")
            elif stripped_response.startswith("```") and stripped_response.endswith("```"):
                 yaml_content_to_parse = stripped_response[3:-3].strip()
                 logger.info("Extracted YAML content by manually stripping ```.")
            else:
                 # No fences found by regex or manual stripping, assume raw YAML
                 logger.warning("No fences found by manual stripping either. Attempting to parse the raw stripped response.")
                 yaml_content_to_parse = stripped_response


        extracted_data = {} # Default to empty dict
        if yaml_content_to_parse:
            try:
                 # Final safety check: Ensure content doesn't start with ``` if stripping failed
                if yaml_content_to_parse.startswith("```"):
                    logger.error("Parsing attempt aborted: Content still starts with ``` after cleanup attempts.")
                    raise yaml.YAMLError("Content started with ``` unexpectedly.")
                    
                extracted_data = yaml.safe_load(yaml_content_to_parse)
                if not isinstance(extracted_data, dict):
                     logger.warning(f"YAML parsing yielded non-dict type: {type(extracted_data)}. Resetting to empty dict.")
                     extracted_data = {}
                logger.info("Successfully parsed extracted YAML content.")
            except yaml.YAMLError as e:
                logger.error(f"YAML parsing failed: {e}. Content that failed (first 500 chars):\n{yaml_content_to_parse[:500]}...")
                extracted_data = {} # Ensure it's an empty dict on error
            except Exception as e:
                 logger.error(f"Unexpected error during YAML parsing: {e}. Content that failed (first 500 chars):\n{yaml_content_to_parse[:500]}...")
                 extracted_data = {}
        else:
            logger.warning("No YAML content could be reliably extracted or parsed from LLM response.")
            extracted_data = {} # Ensure it's an empty dict if no content

        return extracted_data if isinstance(extracted_data, dict) else {}
        # --- END YAML CLEANUP ---

    def generate_soap_note(self, conversation_text: str) -> Dict[str, Any]:
        """
        Generate a SOAP note from conversation text.
        
        Args:
            conversation_text: Text of the conversation
            
        Returns:
            Dictionary with SOAP sections
        """
        template = self.config_manager.get_prompt("soap_note_template")
        if not template:
            template = """
            Generate a comprehensive SOAP note based on the following conversation.

            # SOAP Note Template
            ## Subjective
            - Chief complaint
            - History of present illness
            - Past medical history
            - Medications
            - Allergies
            - Review of systems

            ## Objective
            - Vital signs
            - Physical examination findings
            - Laboratory results
            - Imaging results

            ## Assessment
            - Primary diagnosis
            - Differential diagnoses
            - Clinical reasoning

            ## Plan
            - Diagnostic plans
            - Treatment plans
            - Patient education
            - Follow-up instructions

            Format the response as structured markdown with proper headings and bullet points.
            """
        
        prompt = template + f"\n\nConversation:\n{conversation_text}"
        
        # Configure SOAP note specific parameters
        self.llm_client.set_system_prompt("You are an expert medical documentarian.")
        
        # Process the input and get response
        self.llm_client.add_user_message(text=prompt)
        soap_response = self.llm_client.get_assistant_response(
            max_tokens=800,
            temperature=0.3,
            stream=False
        )
        
        # Parse the SOAP note into sections
        soap_sections = {
            "subjective": "",
            "objective": "",
            "assessment": "",
            "plan": ""
        }
        
        current_section = None
        section_content = []
        
        for line in soap_response.split('\n'):
            if "# SOAP Note" in line or line.strip() == "":
                continue
                
            # Check for section headers
            lower_line = line.lower()
            if "subjective" in lower_line and ('#' in line or '##' in line):
                if current_section and section_content:
                    soap_sections[current_section] = '\n'.join(section_content)
                current_section = "subjective"
                section_content = []
            elif "objective" in lower_line and ('#' in line or '##' in line):
                if current_section and section_content:
                    soap_sections[current_section] = '\n'.join(section_content)
                current_section = "objective"
                section_content = []
            elif "assessment" in lower_line and ('#' in line or '##' in line):
                if current_section and section_content:
                    soap_sections[current_section] = '\n'.join(section_content)
                current_section = "assessment"
                section_content = []
            elif "plan" in lower_line and ('#' in line or '##' in line):
                if current_section and section_content:
                    soap_sections[current_section] = '\n'.join(section_content)
                current_section = "plan"
                section_content = []
            elif current_section:
                section_content.append(line)
        
        # Add the last section
        if current_section and section_content:
            soap_sections[current_section] = '\n'.join(section_content)
            
        # Include full raw note
        soap_sections["full_note"] = soap_response
            
        return soap_sections

    def create_or_update_patient(self, patient_data: Dict[str, Any]) -> Optional[int]:
        """
        Create or update a patient record in the database.
        
        Args:
            patient_data: Dictionary with patient information
            
        Returns:
            Patient ID if successful, None otherwise
        """
        logger.info(f"Creating or updating patient record: {patient_data.get('name', 'Unknown')}")
        
        try:
            # Check if patient exists by name
            name = patient_data.get('name')
            if not name or name == "null":
                logger.warning("Cannot create patient without a name")
                return None
                
            # Check if patient exists by name
            existing_patient = self.db_manager.find_patient_by_name(name)
            
            if existing_patient:
                # Update existing patient
                patient_id = existing_patient['id']
                logger.info(f"Updating existing patient with ID: {patient_id}")
                
                # Extract fields from patient_data
                updated_data = {
                    "name": name,
                    "dob": patient_data.get("dob"),
                    "address": patient_data.get("address"),
                    "phone": patient_data.get("phone"),
                    "email": patient_data.get("email")
                }
                
                # Remove null values
                updated_data = {k: v for k, v in updated_data.items() if v and v != "null"}
                
                # Update the patient
                self.db_manager.update_patient(patient_id, updated_data)
                return patient_id
            else:
                # Create new patient
                logger.info(f"Creating new patient: {name}")
                
                # Extract fields from patient_data
                new_patient_data = {
                    "name": name,
                    "dob": patient_data.get("dob"),
                    "address": patient_data.get("address"),
                    "phone": patient_data.get("phone"),
                    "email": patient_data.get("email")
                }
                
                # Remove null values
                new_patient_data = {k: v for k, v in new_patient_data.items() if v and v != "null"}
                
                # Create the patient
                patient_id = self.db_manager.create_patient(new_patient_data)
                logger.info(f"Created new patient with ID: {patient_id}")
                return patient_id
                
        except Exception as e:
            logger.error(f"Error creating/updating patient: {e}")
            return None


class SpeechAssistant:
    """Speech assistant that coordinates between different AI services."""
    
    def __init__(self):
        """Initialize speech assistant with all available services."""
        self.config_manager = ConfigManager()
        self.scheduler = SchedulerService(self.config_manager)
        self.diagnosis = DifferentialDiagnosisService(self.config_manager)
        self.medication = MedicationReminderService(self.config_manager)
        
        # Start with scheduler as default service
        self.current_service = self.scheduler
        self.conversation_history = []
    
    def set_service(self, service_type: str) -> None:
        """
        Set the current active service.
        
        Args:
            service_type: Type of service to use ('scheduler', 'diagnosis', 'medication')
        """
        if service_type == "scheduler":
            self.current_service = self.scheduler
        elif service_type == "diagnosis":
            self.current_service = self.diagnosis
        elif service_type == "medication":
            self.current_service = self.medication
        logger.info(f"Switched to {service_type} service")
    
    def recognize_speech(self) -> str:
        """
        Recognize speech input.
        
        Returns:
            Transcribed text
        """
        return self.current_service.recognize_speech()
    
    def process_input(self, text_input: str, should_speak: bool = True) -> str:
        """
        Process text input and determine appropriate service.
        
        Args:
            text_input: User input text
            should_speak: Whether to speak the response
            
        Returns:
            Response text
        """
        # Store in conversation history
        self.conversation_history.append({"role": "user", "content": text_input})
        
        # Determine which service to use based on content analysis
        service_type = self._analyze_input(text_input)
        self.set_service(service_type)
        
        # Process with the appropriate service and prompt
        prompt_key = self._get_prompt_key_for_service(service_type)
        response = self.current_service.process_with_prompt(
            prompt_key,
            text_input,
            should_speak=should_speak
        )
        
        # Store response in conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _analyze_input(self, text: str) -> str:
        """
        Analyze input text to determine appropriate service.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Service type ('scheduler', 'diagnosis', 'medication')
        """
        text_lower = text.lower()
        
        # Check for diagnosis keywords
        diagnosis_keywords = ["symptom", "diagnose", "diagnosis", "pain", "feeling", 
                             "hurt", "sick", "ill", "disease", "condition"]
        if any(keyword in text_lower for keyword in diagnosis_keywords):
            return "diagnosis"
        
        # Check for medication keywords
        medication_keywords = ["medication", "medicine", "pill", "drug", "prescription",
                              "dose", "dosage", "refill"]
        if any(keyword in text_lower for keyword in medication_keywords):
            return "medication"
        
        # Default to scheduler for appointment-related or any other queries
        return "scheduler"
    
    def _get_prompt_key_for_service(self, service_type: str) -> str:
        """
        Get the appropriate prompt key for the service type.
        
        Args:
            service_type: Type of service ('scheduler', 'diagnosis', 'medication')
            
        Returns:
            Prompt key from configuration
        """
        mapping = {
            "scheduler": "systemprompt",
            "diagnosis": "differential_system_prompt",
            "medication": "medication_prompt"
        }
        return mapping.get(service_type, "systemprompt")
    
    def save_conversation(self, file_path: str = "conversation.json") -> None:
        """
        Save the conversation history to a file.
        
        Args:
            file_path: Path to save the conversation
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.conversation_history, f, indent=4)
            logger.info(f"Conversation saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
    
    def close(self) -> None:
        """Clean up resources."""
        logger.info("Closing speech assistant")        # Additional cleanup if needed
