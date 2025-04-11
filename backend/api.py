"""
FastAPI application for minoHealth AI services.

This module provides the following endpoints:

Active Endpoints:
1. /tts - Text-to-speech conversion (base64 audio)
2. /tts-stream - Streaming text-to-speech (chunked audio)
3. /transcribe - Speech-to-text transcription
4. /ws/audio - WebSocket for real-time audio streaming
5. /ws/schedule/{patient_id} - WebSocket for appointment scheduling
6. /ws/conversation - WebSocket for medication and appointment conversations
7. /api/medical/extract - Extract structured medical data from transcripts
8. /api/medical/soap - Generate SOAP notes from conversations
9. /health - Health check endpoint

Deprecated Endpoints (to be removed):
1. /api/extract/data - Use /api/medical/extract instead
2. /api/medical/soap - Use /api/medical/soap instead
3. /api/medical/extract - Use /api/medical/extract instead
4. /api/medical/soap - Use /api/medical/soap instead
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query, Depends, HTTPException, status, BackgroundTasks
import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketState
from fastapi.responses import StreamingResponse
import uvicorn
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
import time
import logging
import os
import json
import asyncio
import yaml
import re
import traceback
import base64
import numpy as np
import tempfile
import uuid
import io
from dotenv import load_dotenv
from scipy import signal

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

# Initialize environment variables
load_dotenv()

# Import speech services
from .XTTS_adapter import TTSClient
from .STT_client import SpeechRecognitionClient

# Dynamically import the LLM implementation based on environment variable
llm_implementation = os.getenv("LLM_IMPLEMENTATION", "openrouter").lower()
logger.info(f"Using LLM implementation: {llm_implementation}")

if llm_implementation == "moremi":
    # Import original Moremi implementation
    try:
        from .moremi import ConversationManager
        from .unified_ai_service import (
            AIService,
            ConfigManager,
            SchedulerService,
            DifferentialDiagnosisService,
            MedicationReminderService,
            MedicalAssistantService,
            SpeechAssistant
        )
        logger.info("Successfully loaded original Moremi implementation")
    except ImportError:
        # Fall back to OpenRouter if original implementation is not available
        logger.warning("Original Moremi implementation not found, falling back to OpenRouter")
        from .moremi_open_router import ConversationManager
        from .unified_ai_service_open_router import (
            AIService,
            ConfigManager,
            SchedulerService,
            DifferentialDiagnosisService,
            MedicationReminderService,
            MedicalAssistantService,
            SpeechAssistant
        )
else:
    # Import OpenRouter implementation (default)
    from .moremi_open_router import ConversationManager
    from .unified_ai_service_open_router import (
        AIService,
        ConfigManager,
        SchedulerService,
        DifferentialDiagnosisService,
        MedicationReminderService,
        MedicalAssistantService,
        SpeechAssistant
    )
    logger.info("Successfully loaded OpenRouter implementation")

app = FastAPI(
    title="Moremi AI Scheduler API",
    description="API for scheduling appointments using AI assistance",
    version="1.0.0"
)


# Initialize speech server environment variable
load_dotenv()
speech_server_url = os.getenv("SPEECH_SERVICE_URL")
stt_client = SpeechRecognitionClient(server_url=speech_server_url)
tts_client = TTSClient(api_url=speech_server_url)
logger.info(f"TTS Service URL: {speech_server_url or 'Using default'}")


# Initialize the unified services
config_manager = ConfigManager()
ai_service = AIService(config_manager)
scheduler_service = SchedulerService(config_manager)
diagnosis_service = DifferentialDiagnosisService(config_manager) 
medication_service = MedicationReminderService(config_manager)
medical_assistant_service = MedicalAssistantService(config_manager)

# Log the LLM implementation in use
logger.info(f"Active LLM implementation: {llm_implementation}")
if llm_implementation == "openrouter":
    logger.info(f"Using model: {os.getenv('OPENROUTER_MODEL', 'google/gemini-2.0-flash-lite-001')}")
else:
    logger.info(f"Using model: {os.getenv('MOREMI_MODEL', 'workspace/merged-llava-model')}")

# Class to handle medical data extraction requests
class MedicalExtractionRequest(BaseModel):
    transcript: str
    include_patient_lookup: bool = False

# Class to handle SOAP note generation requests
class SOAPNoteRequest(BaseModel):
    conversation: str

print(f"FastAPI version: {fastapi.__version__}") # Print version at startup

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add this after the FastAPI app initialization
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all incoming requests"""
    start_time = time.time()
    path = request.url.path
    method = request.method
    
    # Only log details for specific endpoints of interest
    detailed_logging = path == "/api/extract/data"
    
    if (detailed_logging):
        logger.info(f"Request started: {method} {path}")
        # Try to log request body for specific endpoints
        try:
            body = await request.body()
            if body:
                logger.info(f"Request body size: {len(body)} bytes")
        except Exception as e:
            logger.warning(f"Could not log request body: {str(e)}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    formatted_process_time = "{0:.2f}".format(process_time)
    
    if detailed_logging:
        logger.info(f"Request completed: {method} {path} - took {formatted_process_time}s with status {response.status_code}")
    
    return response

# TTS request model
class TTSRequest(BaseModel):
    text: str
    speaker: Optional[str] = None

def convert_audio_to_base64(audio_data):
    """Convert numpy array audio data to base64 string."""
    if audio_data is not None:
        # Ensure the array is in the correct format
        audio_bytes = audio_data.tobytes()
        # Convert to base64
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        return base64_audio
    return None

def safe_frombuffer(buffer_data, dtype=np.float32):
    """
    Safely convert buffer to numpy array ensuring it's a multiple of element size.
    
    Args:
        buffer_data (bytes): Binary buffer data
        dtype: NumPy data type (default: np.float32)
        
    Returns:
        np.ndarray: NumPy array with the specified data type
    """
    element_size = np.dtype(dtype).itemsize
    buffer_size = len(buffer_data)
    
    # Check if buffer size is a multiple of element size
    if buffer_size % element_size != 0:
        padding_size = element_size - (buffer_size % element_size)
        logger.info(f"Buffer size {buffer_size} is not a multiple of {element_size}, adding {padding_size} bytes padding")
        # Pad the buffer with zeros to make it a multiple of element size
        padded_buffer = buffer_data + b'\x00' * padding_size
        return np.frombuffer(padded_buffer, dtype=dtype)
    else:
        return np.frombuffer(buffer_data, dtype=dtype)

# Simple API key verification - in production use more secure methods
API_KEYS = {"websocket_key": "audio_access_token_2023"}

async def verify_token(token: str = Query(...)):
    """Simple token verification for WebSocket connections"""
    if token not in API_KEYS.values():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid authentication token",
        )
    return token

@app.get("/")
async def root():
    """Test endpoint to verify API documentation."""
    if llm_implementation == "openrouter":
        return {"message": "Welcome to MinoHealth AI API with OpenRouter integration"}
    else:
        return {"message": "Welcome to MinoHealth AI API with Moremi integration"}

@app.post("/api/medical/extract")
async def extract_medical_data(request: MedicalExtractionRequest):
    """
    Extract structured medical data from a transcript.
    Returns data in YAML format with optional patient database integration.
    """
    try:
        # Log the request
        logger.info(f"Medical data extraction request received: {request.transcript[:50]}...")
        
        # Extract data using the medical assistant service
        extracted_data = medical_assistant_service.extract_data(request.transcript)
        
        # Create or update patient if requested
        patient_id = None
        if request.include_patient_lookup and extracted_data:
            patient_id = medical_assistant_service.create_or_update_patient(extracted_data)
        
        # Format the return value to include both new and old format for compatibility
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        yaml_content = yaml.dump(extracted_data, default_flow_style=False)
        
        # Generate SOAP note
        soap_note = None
        if request.transcript:
            try:
                soap_sections = medical_assistant_service.generate_soap_note(request.transcript)
                # Try to convert the SOAP sections to match expected format
                soap_note = {
                    "SOAP": {
                        "Subjective": {
                            "ChiefComplaint": extracted_data.get("reason_for_visit", ""),
                            "HistoryOfPresentIllness": {
                                "Onset": "",
                                "Location": "",
                                "Duration": "",
                                "Characteristics": "",
                                "AggravatingFactors": "",
                                "RelievingFactors": "",
                                "Timing": "",
                                "Severity": ""
                            },
                            "PastMedicalHistory": "",
                            "FamilyHistory": "",
                            "SocialHistory": "",
                            "ReviewOfSystems": soap_sections.get("subjective", "")
                        },
                        "Assessment": {
                            "PrimaryDiagnosis": extracted_data.get("condition", ""),
                            "DifferentialDiagnosis": "",
                            "ProblemList": ""
                        },
                        "Plan": {
                            "TreatmentAndMedications": "",
                            "FurtherTestingOrImaging": "",
                            "PatientEducation": "",
                            "FollowUp": soap_sections.get("plan", "")
                        }
                    }
                }
            except Exception as e:
                logger.error(f"Error generating SOAP note: {e}")
                soap_note = None
            
        # Create file paths
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        yaml_dir = os.path.join(data_dir, 'yaml_output')
        raw_dir = os.path.join(yaml_dir, 'raw')
        processed_dir = os.path.join(yaml_dir, 'processed')
        soap_dir = os.path.join(data_dir, 'soap')
        
        # Create directories if needed
        for dir_path in [data_dir, yaml_dir, raw_dir, processed_dir, soap_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        raw_path = os.path.join(raw_dir, f'raw_yaml_{timestamp}.yaml')
        processed_path = os.path.join(processed_dir, f'processed_yaml_{timestamp}.yaml')
        soap_path = os.path.join(soap_dir, f'soap_note_{timestamp}.json')
        
        # Save files
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        with open(processed_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        if soap_note:
            with open(soap_path, 'w', encoding='utf-8') as f:
                json.dump(soap_note, f, indent=4)
            
        # Return both new and old formats for compatibility
        return {
            "success": True,
            "status": "success",  # Old format
            "data": extracted_data,
            "patient_id": patient_id,
            "raw_yaml": yaml_content,
            "processed_yaml": yaml_content,
            "files": {
                "raw_yaml": str(raw_path),
                "processed_yaml": str(processed_path),
                "soap_note": str(soap_path) if soap_note else None
            },
            "soap_note": soap_note,
            "message": "Successfully extracted structured data from transcript"
        }
        
    except Exception as e:
        # Log the error
        logger.error(f"Error in medical data extraction: {str(e)}")
        traceback.print_exc()
        return {
            "success": False,
            "status": "error",  # Old format
            "error": str(e)
        }

@app.post("/api/medical/soap")
async def generate_soap_note(request: SOAPNoteRequest):
    """
    Generate a SOAP note from conversation text.
    Returns structured SOAP note with sections.
    """
    try:
        # Log the request
        logger.info(f"SOAP note generation request received: {request.conversation[:50]}...")
        
        # Generate SOAP note using the medical assistant service
        soap_sections = medical_assistant_service.generate_soap_note(request.conversation)
        
        # Extract data from conversation to populate SOAP note format
        extracted_data = medical_assistant_service.extract_data(request.conversation)
        
        # Format in the structure expected by the frontend
        soap_note = {
            "SOAP": {
                "Subjective": {
                    "ChiefComplaint": extracted_data.get("reason_for_visit", ""),
                    "HistoryOfPresentIllness": {
                        "Onset": "",
                        "Location": "",
                        "Duration": "",
                        "Characteristics": "",
                        "AggravatingFactors": "",
                        "RelievingFactors": "",
                        "Timing": "",
                        "Severity": ""
                    },
                    "PastMedicalHistory": "",
                    "FamilyHistory": "",
                    "SocialHistory": "",
                    "ReviewOfSystems": soap_sections.get("subjective", "")
                },
                "Assessment": {
                    "PrimaryDiagnosis": extracted_data.get("condition", ""),
                    "DifferentialDiagnosis": "",
                    "ProblemList": ""
                },
                "Plan": {
                    "TreatmentAndMedications": "",
                    "FurtherTestingOrImaging": "",
                    "PatientEducation": "",
                    "FollowUp": soap_sections.get("plan", "")
                }
            }
        }
        
        # Save the SOAP note to a file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        soap_dir = os.path.join(os.path.dirname(__file__), 'data', 'soap')
        os.makedirs(soap_dir, exist_ok=True)
        soap_path = os.path.join(soap_dir, f'soap_note_{timestamp}.json')

        with open(soap_path, 'w', encoding='utf-8') as f:
            json.dump(soap_note, f, indent=4)
            
        # Return both new and old style responses
        return {
            "success": True,
            "status": "success",  # Old format
            "soap_note": soap_note,
            "raw_sections": soap_sections,  # New format - raw sections
            "file_path": str(soap_path),
            "message": "SOAP note generated successfully"
        }

    except Exception as e:
        # Log the error
        logger.error(f"Error in SOAP note generation: {str(e)}")
        traceback.print_exc()
        return {
            "success": False,
            "status": "error",  # Old format
            "error": str(e)
        }

# Mark deprecated endpoints
@app.post("/api/extract/data", deprecated=True)
async def extract_data(request: Request):
    """
    DEPRECATED: Use /api/medical/extract instead
    Extract structured data from transcript using the medical Assistant
    """
    try:
        data = await request.json()
        transcript = data.get("transcript")
        
        if not transcript:
            raise HTTPException(status_code=400, detail="No transcript provided")
        
        # Use the medical assistant service instead
        extracted_data = medical_assistant_service.extract_data(transcript)
        
        # Create or update patient if data is valid
        patient_id = None
        if extracted_data:
            patient_id = medical_assistant_service.create_or_update_patient(extracted_data)
            
        # Format the return value to match what the frontend expects
        # This ensures compatibility with the existing frontend code
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        yaml_content = yaml.dump(extracted_data, default_flow_style=False)
        
        # Generate SOAP note
        soap_note = None
        if transcript:
            try:
                soap_sections = medical_assistant_service.generate_soap_note(transcript)
                # Try to convert the SOAP sections to match expected format
                soap_note = {
                    "SOAP": {
                        "Subjective": {
                            "ChiefComplaint": extracted_data.get("reason_for_visit", ""),
                            "HistoryOfPresentIllness": {
                                "Onset": "",
                                "Location": "",
                                "Duration": "",
                                "Characteristics": "",
                                "AggravatingFactors": "",
                                "RelievingFactors": "",
                                "Timing": "",
                                "Severity": ""
                            },
                            "PastMedicalHistory": "",
                            "FamilyHistory": "",
                            "SocialHistory": "",
                            "ReviewOfSystems": soap_sections.get("subjective", "")
                        },
                        "Assessment": {
                            "PrimaryDiagnosis": extracted_data.get("condition", ""),
                            "DifferentialDiagnosis": "",
                            "ProblemList": ""
                        },
                        "Plan": {
                            "TreatmentAndMedications": "",
                            "FurtherTestingOrImaging": "",
                            "PatientEducation": "",
                            "FollowUp": soap_sections.get("plan", "")
                        }
                    }
                }
            except Exception as e:
                logger.error(f"Error generating SOAP note: {e}")
                soap_note = None
            
        # Create file paths (these are fictional but help with frontend compatibility)
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        yaml_dir = os.path.join(data_dir, 'yaml_output')
        raw_dir = os.path.join(yaml_dir, 'raw')
        processed_dir = os.path.join(yaml_dir, 'processed')
        soap_dir = os.path.join(data_dir, 'soap')
        
        # Create directories if needed
        for dir_path in [data_dir, yaml_dir, raw_dir, processed_dir, soap_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        raw_path = os.path.join(raw_dir, f'raw_yaml_{timestamp}.yaml')
        processed_path = os.path.join(processed_dir, f'processed_yaml_{timestamp}.yaml')
        soap_path = os.path.join(soap_dir, f'soap_note_{timestamp}.json')
        
        # Save files
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        with open(processed_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        if soap_note:
            with open(soap_path, 'w', encoding='utf-8') as f:
                json.dump(soap_note, f, indent=4)
        
        # Return the format expected by the frontend
            return {
                "status": "success",
            "data": extracted_data,
                "patient_id": patient_id,
            "raw_yaml": yaml_content,
            "processed_yaml": yaml_content,
                "files": {
                    "raw_yaml": str(raw_path),
                    "processed_yaml": str(processed_path),
                "soap_note": str(soap_path) if soap_note else None
                },
            "soap_note": soap_note,
                "message": "Successfully extracted structured data from transcript"
            }
            
    except Exception as e:
        logger.error(f"Error in extract_data endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Server error during data extraction: {str(e)}"
        )

@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket, token: str = Query(None)):
    await websocket.accept()
    logger.info("Audio WebSocket connection established")
    
    try:
        while True:
            try:
                data = await websocket.receive_json()
                message_type = data.get("type")
                payload = data.get("payload", {})
                
                if message_type == "transcription":
                    try:
                        audio_data = payload.get("audioData")
                        audio_format = payload.get("format", "webm")
                        codec = payload.get("codec", "opus")
                        
                        if not audio_data:
                            await websocket.send_json({
                                "type": "transcription",
                                "text": "No audio data provided"
                            })
                            continue
                        
                        # Decode base64 audio data
                        audio_bytes = base64.b64decode(audio_data)
                        
                        if audio_format == "webm" and codec == "opus":
                            # For WebM/Opus, we need to decode it first
                            import soundfile as sf
                            import io
                            
                            # Create a temporary file to save the WebM audio
                            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                                temp_file.write(audio_bytes)
                                temp_file_path = temp_file.name
                            
                            try:
                                # Read the WebM file using soundfile
                                with sf.SoundFile(temp_file_path) as f:
                                    audio_np = f.read()
                                    sample_rate = f.samplerate
                                    
                                # Convert to mono if stereo
                                if len(audio_np.shape) > 1:
                                    audio_np = np.mean(audio_np, axis=1)
                                
                                # Resample to 16kHz if needed
                                if sample_rate != 16000:
                                    num_samples = int(len(audio_np) * 16000 / sample_rate)
                                    audio_np = signal.resample(audio_np, num_samples)
                                
                                # Normalize audio
                                if len(audio_np) > 0:
                                    max_val = np.max(np.abs(audio_np))
                                    if max_val > 0:
                                        audio_np = audio_np / max_val
                                
                                # Convert to float32
                                audio_np = audio_np.astype(np.float32)
                            finally:
                                # Clean up temporary file
                                os.unlink(temp_file_path)
                        else:
                            # For raw PCM float32 data
                            audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                        
                        # Check if audio contains actual data
                        if len(audio_np) == 0 or np.all(np.abs(audio_np) < 1e-6):
                            logger.warning("Audio data contains all zeros or is too quiet")
                            await websocket.send_json({
                                "type": "transcription",
                                "text": "No speech detected"
                            })
                            continue
                        
                        # Use the client's transcribe_audio method
                        transcription = stt_client.transcribe_audio(audio_np.tobytes())
                        logger.info(f"Transcription result: {transcription}")
                        
                        if transcription and not transcription.startswith("Processing error"):
                            await websocket.send_json({
                                "type": "transcription",
                                "text": transcription
                            })
                        else:
                            await websocket.send_json({
                                "type": "transcription",
                                "text": "No speech detected"
                            })
                            
                    except Exception as e:
                        logger.error(f"Error processing audio data: {str(e)}")
                        if websocket.client_state == WebSocketState.CONNECTED:
                            await websocket.send_json({
                                "type": "error",
                                "text": f"Error processing audio: {str(e)}"
                            })
                            
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
            except Exception as e:
                logger.error(f"Error in websocket communication: {str(e)}")
                break
                
    except Exception as e:
        logger.error(f"Error in audio websocket: {str(e)}")
    finally:
        logger.info("Audio WebSocket connection closed")

@app.websocket("/ws/schedule/{patient_id}")
async def schedule_session(websocket: WebSocket, patient_id: int):
    """
    WebSocket endpoint that handles the entire scheduling flow
    """
    logger.info(f"Starting scheduling session for patient {patient_id}")
    await websocket.accept()
    
    try:
        # Initialize using unified service architecture
        assistant = SpeechAssistant()
        assistant.patient_id = patient_id
        
        # Import the database manager within the function to avoid circular imports
        from .database_utils import PatientDatabaseManager
        db_manager = PatientDatabaseManager()
        
        # Verify patient exists
        patient = db_manager.get_patient_by_id(patient_id)
        if not patient:
            logger.warning(f"Patient with ID {patient_id} not found, continuing with default patient data")
            # Instead of returning an error, continue with a default approach
            patient = {
                "patient_id": patient_id,
                "name": "Patient",
                "phone": "",
                "email": "",
                "address": "",
                "date_of_birth": None
            }
        
        # Send initial greeting
        greeting = "Welcome to Moremi AI Scheduler! Please hold while I process your conversation."
        await websocket.send_json({
            "type": "message",
            "text": greeting
        })
        
        tts_client.TTS(
            greeting,
            play_locally=True
        ) 
        tts_client.wait_for_completion()

        while True:
            try:
                message = await websocket.receive_json()
                logger.debug(f"Received message type: {message.get('type')}")
                
                if message.get("type") == "context":
                    # Process initial context
                    context = message.get("context", "")
                    assistant.LLM.add_user_message(text=assistant.manage_context(context))
                    response = assistant.LLM.get_assistant_response()
                    
                    # Process response for doctor/specialty search
                    assistant.process_initial_response(response)
                    
                    # Send the status messages to the frontend if available
                    if hasattr(assistant, 'status_messages') and assistant.status_messages:
                        for status_msg in assistant.status_messages:
                            if status_msg:
                                await websocket.send_json({
                                    "type": "message",
                                    "role": "system",
                                    "text": status_msg
                                })
                                
                                tts_client.TTS(
                                    status_msg,
                                    play_locally=True
                                ) 
                                tts_client.wait_for_completion()
                        
                    
                    # Handle available slots
                    if assistant.suit_doc:
                        slots_response = ""
                        for doc in assistant.suit_doc:
                            # Use the unified scheduler service to get slots
                            slots = db_manager.get_available_slots(doctor_id=doc['doctor_id'], days_ahead=30)
                            if slots:
                                slots_response += f'\n{db_manager.format_available_slots(slots, doc["doctor_name"])}\n'
                                break
                            elif db_manager.days_ahead == 90:
                                slots_response += f'\n{db_manager.format_available_slots(slots, doc["doctor_name"])}\n'
                                break
                            else:
                                db_manager.days_ahead += 5
                        
                        await websocket.send_json({
                            "type": "message",
                            "text": slots_response
                        })
                        
                        tts_client.TTS(
                            slots_response,
                            play_locally=True
                        )
                        
                        tts_client.wait_for_completion()
                    else:
                        await websocket.send_json({
                            "type": "message",
                            "text": "No doctors available at this time. Please try again later."
                        })
                        
                        tts_client.TTS(
                            "No doctors available at this time. Please try again later.",
                            play_locally=True
                        )
                        tts_client.wait_for_completion()

                elif message.get("type") == "message":
                    # Process user message
                    user_input = message.get("text", "")
                    logger.debug(f"Processing user input: {user_input[:50]}...")
                    
                    assistant.LLM.add_user_message(text=user_input)
                    response = assistant.LLM.get_assistant_response(should_speak=True)
                    
                    await websocket.send_json({
                        "type": "message",
                        "text": response
                    })

                elif message.get("type") == "end_conversation":
                    logger.info("Ending conversation and generating summary")
                    # Change to summary system prompt
                    assistant.LLM.custom_params["system_prompt"] = assistant.summary_system_prompt
                    assistant.LLM.add_user_message(text="Extract the agreed upon schedule from the interaction history. If there was no agreed upon appointment respond with the word null.")
                    summary = assistant.LLM.get_assistant_response()
                    
                    if summary and summary.lower() != "null":
                        try:
                            parsed = json.loads(summary)
                            # Save appointment details using the scheduler service
                            appointment_id = scheduler_service.create_appointment(
                                patient_id=patient_id,
                                appointment_details=parsed
                            )
                            
                            # Add appointment ID to the response
                            parsed["appointment_id"] = appointment_id
                            
                            await websocket.send_json({
                                "type": "summary",
                                "text": json.dumps(parsed)
                            })
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse appointment summary: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "message": "Failed to process appointment details"
                            })
                    else:
                        await websocket.send_json({
                            "type": "summary",
                            "text": "null"
                        })
                    break

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {str(e)}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected during setup")
    except Exception as e:
        logger.error(f"Error in schedule_session: {str(e)}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    finally:
        logger.info(f"Cleaning up schedule_session for patient {patient_id}")
        tts_client.stop()

@app.websocket("/ws/conversation")
async def conversation_websocket(websocket: WebSocket, patient_id: int = Query(...), type: str = Query(...), token: str = Query(None)):
    """
    WebSocket endpoint for medication and appointment conversations
    """
    # Check authentication token
    if token not in API_KEYS.values():
        logger.warning(f"Authentication failed for WebSocket connection: {token}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
        
    try:
        await websocket.accept()
        logger.info(f"Conversation WebSocket connection established for patient {patient_id} with type {type}")
        
        # Track conversation state to prevent duplicate messages
        conversation_state = {
            "last_message_id": None,  # Use message ID to track uniqueness
            "last_message": None,
            "message_count": 0,
            "audio_format": None,  # Track audio format to ensure consistency
            "greeting_sent": False  # Flag to track if greeting has been sent
        }
        
        # Initialize dictionaries for managing active conversations if they don't exist
        if not hasattr(app.state, "active_conversations"):
            app.state.active_conversations = {}
        
        # Look up the reminder if a reminder_id is provided
        reminder_id = f"{type}_{patient_id}"
        reminder = None
        
        # Directly access the reminder from the dictionary using the key
        reminder = app.state.active_conversations.get(reminder_id)
        
        # If not found with simple key, try to find it in all values
        if not reminder:
            for key, value in app.state.active_conversations.items():
                if key.startswith(f"{type}_{patient_id}"):
                    reminder = value
                    break
        
        logger.info(f"Reminder lookup for {reminder_id}: {'Found' if reminder else 'Not found'}")
        
        # Initialize the appropriate service from unified_ai_service.py
        service = None
        try:
            # Create a dummy data class to hold patient information
            class ReminderData:
                def __init__(self, patient_id, reminder_type):
                    self.patient_id = patient_id
                    self.message_type = reminder_type
                    self.patient_name = "Patient"  # Default name
                    self.details = {"patient_id": patient_id}
            
            # Use reminder data if available, otherwise create placeholder
            reminder_data = reminder or ReminderData(patient_id, type)
            
            # Initialize the appropriate service based on type
            if type == "medication":
                service = medication_service
            elif type == "appointment":
                service = scheduler_service
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Invalid conversation type: {type}",
                    "message_id": "error_invalid_type"
                })
                return
                
        except Exception as agent_err:
            logger.error(f"Failed to initialize service: {str(agent_err)}")
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to initialize conversation service: {str(agent_err)}",
                "message_id": "error_service_init"
            })
            return
            
        # Send initial message only if we haven't sent a greeting yet
        if not conversation_state["greeting_sent"]:
            # Customize greeting based on conversation type
            if type == "medication":
                initial_message = f"Hello {getattr(reminder, 'patient_name', 'Patient')}! This is MinoHealth calling about your medication reminder. How are you doing today?"
            elif type == "appointment":
                initial_message = f"Hello {getattr(reminder, 'patient_name', 'Patient')}! This is MinoHealth calling about your upcoming appointment. How are you doing today?"
            else:
                initial_message = f"Hello {getattr(reminder, 'patient_name', 'Patient')}! This is MinoHealth calling. How can I assist you today?"
                
            message_id = f"initial_greeting_{patient_id}_{type}"
            
            try:
                await websocket.send_json({
                    "type": "message",
                    "text": initial_message,
                    "message_id": message_id
                })
                conversation_state["message_count"] += 1
                conversation_state["last_message"] = initial_message
                conversation_state["last_message_id"] = message_id
                conversation_state["greeting_sent"] = True
                
                # Try to generate speech for the initial message using unified service
                audio_info = tts_client.get_audio_for_frontend(initial_message)
                
                if audio_info and 'audio' in audio_info:
                    # Store audio format information
                    conversation_state["audio_format"] = {
                        "sample_rate": audio_info.get('sample_rate', 24000),
                        "encoding": "PCM_FLOAT"
                    }
                    
                    await websocket.send_json({
                        "type": "audio",
                        "audio": audio_info['audio'],
                        "sample_rate": audio_info.get('sample_rate', 24000),
                        "message_id": f"audio_{message_id}",
                        "encoding": "PCM_FLOAT"
                    })
                    
                    # Send a prompt to encourage user input
                    await websocket.send_json({
                        "type": "status",
                        "status": "waiting_for_input",
                        "message": "Please speak now",
                        "message_id": "prompt_for_input"
                    })
            except Exception as audio_err:
                logger.error(f"Failed to generate speech for initial message: {str(audio_err)}")
        
        # Handle audio input from client
        while True:
            try:
                data_raw = await websocket.receive_text()
                data = json.loads(data_raw)
                
                if "type" not in data:
                    logger.warning(f"Received message without type: {data}")
                    continue
                    
                message_type = data["type"]
                
                # Handle audio input
                if message_type == "audio_input":
                    audio_base64 = data.get("audio")
                    conversation_id = data.get("conversation_id")
                    client_message_id = data.get("message_id", f"audio_input_{datetime.now().timestamp()}")
                    
                    if not audio_base64:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No audio data provided",
                            "message_id": f"error_no_audio_{client_message_id}"
                        })
                        continue
                    
                    # Process audio for transcription
                    try:
                        audio_bytes = base64.b64decode(audio_base64)
                        # Use the stt_client from the unified service
                        transcription = stt_client.transcribe_audio(audio_bytes)
                        
                        # Check if transcription contains an error message
                        if transcription and transcription.startswith("Processing error:"):
                            logger.error(f"Transcription error: {transcription}")
                            await websocket.send_json({
                                "type": "error",
                                "message": transcription,
                                "message_id": f"error_transcription_{client_message_id}"
                            })
                            continue
                        
                        # Generate a unique ID for this transcription
                        transcription_id = f"transcription_{conversation_state['message_count']}_{datetime.now().timestamp()}"
                        
                        if not transcription or transcription == "No speech detected":
                            await websocket.send_json({
                                "type": "message",
                                "text": "I couldn't hear you clearly. Could you please repeat that?",
                                "message_id": f"no_speech_{transcription_id}"
                            })
                            continue
                            
                        # Send transcription back to client
                        await websocket.send_json({
                            "type": "message",
                            "role": "user",  # Add role to identify user messages
                            "text": transcription
                        })
                        
                        # Process with the appropriate unified service
                        if service:
                            # Generate response based on conversation type
                            if type == "medication":
                                response = service.handle_medication_reminder({
                                    "patient_name": getattr(reminder, "patient_name", "Patient"),
                                    "patient_id": patient_id,
                                    "message": transcription
                                })
                            elif type == "appointment":
                                response = service.handle_appointment_reminder({
                                    "patient_name": getattr(reminder, "patient_name", "Patient"),
                                    "patient_id": patient_id,
                                    "message": transcription
                                })
                            else:
                                # Default to simple processing with base AI service
                                response = ai_service.process_with_prompt(
                                    "conversation", 
                                    transcription
                                )
                            
                            # Send text response
                            await websocket.send_json({
                                "type": "message",
                                "role": "assistant",  # Add role to identify assistant messages
                                "text": response
                            })
                            
                            # Generate and send audio
                            audio_info = tts_client.get_audio_for_frontend(response)
                            
                            if audio_info and 'audio' in audio_info:
                                await websocket.send_json({
                                    "type": "audio",
                                    "audio": audio_info['audio'],
                                    "sample_rate": audio_info.get('sample_rate', 24000)
                                })
                        else:
                            # Fallback response if service isn't properly initialized
                            response = f"Thank you for your message: '{transcription}'. How else can I help with your {type}?"
                            response_id = f"fallback_{conversation_state['message_count']}_{hash(response)}"
                            
                            # Send text response
                            await websocket.send_json({
                                "type": "message",
                                "role": "assistant",  # Add role to identify assistant messages
                                "text": response
                            })
                            
                            # Try to generate speech
                            try:
                                audio_info = tts_client.get_audio_for_frontend(response)
                                
                                if audio_info and 'audio' in audio_info:
                                    await websocket.send_json({
                                        "type": "audio",
                                        "audio": audio_info['audio'],
                                        "sample_rate": audio_info.get('sample_rate', 24000)
                                    })
                            except Exception as audio_err:
                                logger.error(f"Failed to generate speech for response: {str(audio_err)}")
                                
                    except Exception as transcribe_err:
                        logger.error(f"Error processing audio: {str(transcribe_err)}")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Failed to process your audio",
                            "message_id": f"error_process_{datetime.now().timestamp()}"
                        })
                
                # Handle ping to keep connection alive
                elif message_type == "ping":
                    timestamp = datetime.now().timestamp()
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": timestamp,
                        "message_id": f"pong_{timestamp}"
                    })
                
                # Handle commands
                elif message_type == "command":
                    command = data.get("command")
                    command_id = f"cmd_{command}_{datetime.now().timestamp()}"
                    
                    if command == "finish":
                        farewell = "Thank you for using our service. Goodbye!"
                        await websocket.send_json({
                            "type": "message",
                            "text": farewell,
                            "message_id": f"farewell_{command_id}"
                        })
                        
                        # Generate farewell audio
                        try:
                            audio_info = tts_client.get_audio_for_frontend(farewell)
                            
                            if audio_info and 'audio' in audio_info:
                                await websocket.send_json({
                                    "type": "audio",
                                    "audio": audio_info['audio'],
                                    "sample_rate": audio_info.get('sample_rate', 24000),
                                    "message_id": f"audio_farewell_{command_id}",
                                    "encoding": "PCM_FLOAT"
                                })
                        except Exception as e:
                            logger.error(f"Error generating farewell audio: {e}")
                            
                        break
                    else:
                        await websocket.send_json({
                            "type": "status",
                            "message": f"Received command: {command}",
                            "message_id": command_id
                        })
                
                # Handle other message types
                else:
                    message_id = f"unknown_{message_type}_{datetime.now().timestamp()}"
                    await websocket.send_json({
                        "type": "status",
                        "message": f"Received unknown message type: {message_type}",
                        "message_id": message_id
                    })
                    
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from WebSocket message")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format received",
                    "message_id": f"error_json_{datetime.now().timestamp()}"
                })
                continue
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for patient {patient_id}")
                break
            except Exception as e:
                logger.error(f"Error in conversation WebSocket: {str(e)}")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Server error: {str(e)}",
                        "message_id": f"error_server_{datetime.now().timestamp()}"
                    })
                except:
                    # If we can't send the error, the connection is probably closed
                    break
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for patient {patient_id}")
    except Exception as e:
        logger.error(f"Error in conversation WebSocket: {str(e)}")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

@app.websocket("/ws/diagnosis/{patient_id}")
async def diagnosis_session(websocket: WebSocket, patient_id: int):
    """Handle a diagnosis conversation session."""
    try:
        await websocket.accept()  # Accept the WebSocket connection first
        
        # Import the database manager within the function to avoid circular imports
        from .database_utils import PatientDatabaseManager
        db_manager = PatientDatabaseManager()
        
        # Verify patient exists
        patient = db_manager.get_patient_by_id(patient_id)
        if not patient:
            logger.warning(f"Patient with ID {patient_id} not found, continuing with default patient data")
            patient = {
                "patient_id": patient_id,
                "name": "Patient",
                "phone": "",
                "email": "",
                "address": "",
                "date_of_birth": None
            }
        
        # Initialize conversation history with system message
        conversation_history = [
            {"role": "system", "content": "You are a helpful medical diagnosis assistant."}
        ]
        
        # Add greeting to conversation history only
        greeting = f"Hello! I'm your differential diagnosis assistant. How can I help you today?"
        conversation_history.append({"role": "agent", "content": greeting})
        
        # Send initial greeting
        await websocket.send_json({
            "type": "message",
            "role": "agent",
            "text": greeting
        })
        
        tts_client.TTS(
            greeting,
            play_locally=True
        )
        
        tts_client.wait_for_completion()
        
        # Process messages
        while True:
            try:
                data = await websocket.receive_json()
                message_type = data.get("type")
                
                # Handle keepalive pings
                if message_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                if message_type == "end_conversation":
                    logger.info("Received end_conversation signal. Generating summary...")
                    conversation_text = data.get("conversation", "") # Get conversation text sent by frontend

                    if conversation_text:
                        # Call the appropriate service method
                        try:
                            # Use the diagnosis_service instance already created
                            # This call uses the 'differential_summary_systemprompt' via process_with_prompt
                            summary_result = diagnosis_service.summarize_diagnosis(conversation_text)
                            logger.info(f"Generated summary result (raw): {summary_result[:200]}...") # Log beginning of summary

                            # Send the summary back to the client
                            # The frontend expects the summary data in the 'text' field
                            # The frontend handler will parse this if it's a JSON string
                            await websocket.send_json({
                                "type": "diagnosis_summary", # Use the correct type expected by frontend
                                "text": summary_result # Send the raw result (expected JSON string or object)
                            })
                            logger.info("Sent diagnosis_summary message to client.")

                        except Exception as summary_err:
                            logger.error(f"Error during diagnosis summary generation: {summary_err}", exc_info=True)
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Failed to generate diagnosis summary: {str(summary_err)}"
                            })
                    else:
                         logger.warning("Received end_conversation but no conversation text was provided.")
                         await websocket.send_json({
                                "type": "error",
                                "message": "No conversation text received to generate summary."
                            })
                    break # End the loop after handling end_conversation

                if message_type == "message":
                    user_message = data.get("text", "")
                    if not user_message:
                        continue
                    
                    # Add user message to history
                    conversation_history.append({"role": "user", "content": user_message})
                    
                    # Process with diagnosis service
                    response = diagnosis_service.diagnose(user_message)
                    
                    # Add assistant response to history
                    conversation_history.append({"role": "agent", "content": response})
                                        
                    # Send response to client
                    await websocket.send_json({
                        "type": "message",
                        "role": "agent",
                        "text": response
                    })
                    
                    tts_client.TTS(
                        response,
                        play_locally=True
                    )
                    
                    tts_client.wait_for_completion()
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                await websocket.send_json({
                      "type": "error",
                    "message": "Error processing message. Please try again."
                })
                
            except Exception as e:
                logger.error(f"Error in diagnosis session: {str(e)}")
            # if websocket.client_state == WebSocketState.CONNECTED:  # Fixed client state check
            #     await websocket.send_json({
            #         "type": "error",
            #         "message": "An error occurred. Please try again."
            #     })
    finally:
        try:
            await websocket.close()
        except:
            pass

async def stream_diagnosis_response(user_input: str):
    """
    Simulate streaming response from the diagnosis service.
    In a real implementation, this would connect to a streaming LLM API.
    
    Args:
        user_input: User's message
    
    Yields:
        Partial response strings
    """
    try:
        # Get the full response first
        full_response = diagnosis_service.diagnose(user_input)
        
        # Split into sentences or chunks
        chunks = re.split(r'(?<=[.!?]) +', full_response)
        
        partial = ""
        for chunk in chunks:
            partial += chunk + " "
            yield partial.strip()
            await asyncio.sleep(0.1)  # Simulate network delay
    except Exception as e:
        logger.error(f"Error in streaming diagnosis response: {str(e)}")
        yield f"I'm sorry, I encountered an error processing your request. {str(e)}"

async def stream_diagnosis_summary(conversation: str):
    """
    Simulate streaming summary from the diagnosis service.
    
    Args:
        conversation: Full conversation history
    
    Yields:
        Partial summary strings
    """
    try:
        # Get the full summary first
        full_summary = diagnosis_service.summarize_diagnosis(conversation)
        
        try:
            # If it's valid JSON, we can stream it in progressive steps
            summary_obj = json.loads(full_summary)
            
            # Stream in logical steps
            steps = [
                {"status": "Processing primary diagnosis..."},
                {"primaryDiagnosis": summary_obj.get("primaryDiagnosis", "")},
                {"primaryDiagnosis": summary_obj.get("primaryDiagnosis", ""), 
                 "differentialDiagnoses": summary_obj.get("differentialDiagnoses", [])},
                {"primaryDiagnosis": summary_obj.get("primaryDiagnosis", ""), 
                 "differentialDiagnoses": summary_obj.get("differentialDiagnoses", []),
                 "recommendedTests": summary_obj.get("recommendedTests", [])},
                summary_obj  # Final complete object
            ]
            
            for step in steps:
                yield json.dumps(step)
                await asyncio.sleep(0.3)  # Slightly longer delay for summary
                
        except json.JSONDecodeError:
            # Fall back to simpler approach for non-JSON responses
            yield '{"status": "Processing..."}'
            await asyncio.sleep(0.5)
            yield full_summary
    except Exception as e:
        logger.error(f"Error in streaming diagnosis summary: {str(e)}")
        yield json.dumps({"error": f"Error generating summary: {str(e)}"})

@app.get("/llm-config")
async def get_llm_config():
    """Get the current LLM configuration details."""
    if llm_implementation == "openrouter":
        return {
            "implementation": "openrouter",
            "model": os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-lite-001"),
            "api_base": "https://openrouter.ai/api/v1",
            "site_info": {
                "site_url": os.getenv("SITE_URL", ""),
                "site_name": os.getenv("SITE_NAME", "CRM")
            }
        }
    else:
        return {
            "implementation": "moremi",
            "model": os.getenv("MOREMI_MODEL", "workspace/merged-llava-model"),
            "api_base": os.getenv("MOREMI_API_BASE_URL", "")
        }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)