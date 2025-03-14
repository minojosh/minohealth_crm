from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query, Depends, HTTPException, status, BackgroundTasks
from fastapi.websockets import WebSocketState
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from .scheduler import (
    SpeechAssistant, Message, docs, db_manager,
    speech_client, schedule_appointment
)
from .medical_assistant import Assistant
# Import appointment manager components
from .appointment_manager import (
    SchedulerManager, 
    MedicalAgent, 
    ReminderMessage
)
from datetime import datetime
import fastapi  # Import the fastapi module itself
from typing import Optional, List
from pydantic import BaseModel
# from .TTS_client import TTSClient
from .XTTS_adapter import TTSClient
# from .XTTS_client import TTSClient
import time
import logging
import os
import json
import asyncio
import fastapi
import yaml
import re
import traceback
import base64
import numpy as np
from typing import Optional, List
from pydantic import BaseModel
from .STT_client import SpeechRecognitionClient
from fastapi.websockets import WebSocketState
import os
import tempfile
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

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path, encoding='utf-8')

app = FastAPI(
    title="Moremi AI Scheduler API",
    description="API for scheduling appointments using AI assistance",
    version="1.0.0"
)


# Initialize STT client with URL from environment variable
stt_server_url = os.getenv("STT_SERVER_URL")
stt_client = SpeechRecognitionClient(server_url=stt_server_url)
logger.info(f"STT Service URL: {stt_server_url or 'Using default'}")

# Initialize TTS client
tts_client = TTSClient(api_url=os.getenv("XTTS_SERVER_URL"))
logger.info(f"TTS Service URL: {os.getenv('XTTS_SERVER_URL') or 'Using default'}")

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

# Appointment Manager API models
class SchedulerRequest(BaseModel):
    hours_ahead: int
    days_ahead: int
    type: str  # 'appointment' or 'medication'

class ConversationResponse(BaseModel):
    status: str
    message: str
    conversation_id: Optional[str]

class ReminderResponse(BaseModel):
    patient_name: str
    message_type: str
    details: dict
    timestamp: datetime

# TTS request model
class TTSRequest(BaseModel):
    text: str
    speaker: Optional[str] = None

# Store active conversations
active_conversations = {}
# Store last audio data for each conversation
conversation_audio_cache = {}


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
    return {"message": "Welcome to Moremi AI Scheduler API"}

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Generate speech from text using the TTS client.
    Returns base64-encoded audio data.
    """
    try:
        logger.info(f"TTS request received for text: {request.text[:30]}...")
        
        # Use the TTS client to generate audio - correctly unpack all 4 return values
        audio_data, sample_rate, base64_audio, _ = tts_client.TTS(
            request.text,
            play_locally=False,  # Don't play on server
            return_data=True     # Get the data back
        )
        
        return {
            "audio": base64_audio,
            "sample_rate": sample_rate,
            "text": request.text
        }
    except Exception as e:
        logger.error(f"Error in TTS endpoint: {str(e)}")
        return {"error": str(e)}

@app.post("/transcribe")
async def transcribe_audio_endpoint(request: Request):
    """
    HTTP endpoint for audio transcription.
    This proxies requests to the external STT service.
    """
    try:
        data = await request.json()
        
        # Check if this is a finish command or audio data
        if 'command' in data and data['command'] == 'finish':
            logger.info("Finish command received, proxying to STT service")
            return {"transcription": "", "status": "finished"}
        
        if 'audio' in data:
            # Process audio data
            audio_data = data['audio']
            try:
                # Log incoming data stats
                logger.info(f"Received base64 audio data of length: {len(audio_data)}")
                
                # Convert from base64 to bytes
                audio_bytes = base64.b64decode(audio_data)
                logger.info(f"Decoded audio bytes length: {len(audio_bytes)}")
                
                # Convert to numpy array for analysis
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                audio_stats = {
                    "length": len(audio_np),
                    "duration": len(audio_np) / 16000,  # Assuming 16kHz sample rate
                    "min": float(np.min(audio_np)),
                    "max": float(np.max(audio_np)),
                    "mean": float(np.mean(audio_np)),
                    "std": float(np.std(audio_np)),
                    "non_zero": int(np.count_nonzero(audio_np))
                }
                logger.info(f"Audio statistics: {audio_stats}")
                
                # Process using the STT client
                logger.info("Sending audio to STT service...")
                transcription = stt_client.transcribe_audio(audio_bytes)
                
                logger.info(f"Transcription result: {transcription}")
                return {"transcription": transcription}
            except Exception as e:
                logger.error(f"Error processing audio data: {str(e)}", exc_info=True)
                return {"error": f"Failed to process audio: {str(e)}"}
        
        return {"error": "Invalid request format"}
    
    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {str(e)}", exc_info=True)
        return {"error": f"Internal server error: {str(e)}"}

@app.post("/api/extract/audio")
async def extract_from_audio(request: Request):
    """Extract structured data directly from audio using the medical Assistant"""
    try:
        data = await request.json()
        audio_base64 = data.get("audio")
        
        if not audio_base64:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        # Decode audio data
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            logger.error(f"Error decoding audio data: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid audio data format")
        
        # Transcribe audio
        try:
            transcript =stt_client.transcribe_audio(audio_bytes)
            
            if not transcript or transcript == "No speech detected":
                raise HTTPException(status_code=400, detail="No speech detected in audio")
                
            logger.info(f"Transcription result: {transcript}")
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")
        
        # Use the Assistant class from medical_assistant.py
        from .medical_assistant import Assistant, Message
        
        # Create Assistant and add transcript
        assistant = Assistant()
        assistant.messages.append(Message(user_input=transcript))
        
        # Extract structured YAML using Moremi
        try:
            # Get YAML response from Moremi
            yaml_content = assistant.get_moremi_response(assistant.messages, assistant.system_prompt)
            timestamp = assistant._get_timestamp()
            
            # Save YAML files
            raw_path, processed_path = assistant._save_yaml(yaml_content, timestamp)
            
            # Parse YAML
            import yaml
            processed_yaml = yaml.safe_load(yaml_content)
            
            # Try to create a patient record if data is valid
            patient_id = None
            if processed_yaml and isinstance(processed_yaml, dict):
                patient_id = assistant._create_patient_from_yaml(processed_yaml)
            
            # Return the structured data along with file paths and patient ID
            return {
                "status": "success",
                "transcript": transcript,
                "data": processed_yaml,
                "patient_id": patient_id,
                "files": {
                    "raw_yaml": str(raw_path),
                    "processed_yaml": str(processed_path)
                },
                "message": "Successfully extracted structured data from audio"
            }
        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Error extracting data from transcript: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in extract_from_audio endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Server error during audio data extraction: {str(e)}"
        )

@app.post("/api/extract/data")
async def extract_data(request: Request):
    """Extract structured data from transcript using the medical Assistant"""
    try:
        data = await request.json()
        transcript = data.get("transcript")
        
        if not transcript:
            raise HTTPException(status_code=400, detail="No transcript provided")
        
        # Use the Assistant class from medical_assistant.py
        from .medical_assistant import Assistant, Message
        
        # Create Assistant and add transcript
        assistant = Assistant()
        assistant.messages.append(Message(user_input=transcript))
        
        # Extract structured YAML using Moremi
        try:
            # Get YAML response from Moremi
            yaml_content = assistant.get_moremi_response(assistant.messages, assistant.system_prompt)
            timestamp = assistant._get_timestamp()
            
            # Save YAML files
            raw_path, processed_path = assistant._save_yaml(yaml_content, timestamp)
            
            # Parse YAML
            import yaml
            processed_yaml = yaml.safe_load(yaml_content)
            
            # Try to create a patient record if data is valid
            patient_id = None
            if processed_yaml and isinstance(processed_yaml, dict):
                patient_id = assistant._create_patient_from_yaml(processed_yaml)
            
            # Return the structured data along with file paths and patient ID
            return {
                "status": "success",
                "data": processed_yaml,
                "patient_id": patient_id,
                "files": {
                    "raw_yaml": str(raw_path),
                    "processed_yaml": str(processed_path)
                },
                "message": "Successfully extracted structured data from transcript"
            }
        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Error extracting data from transcript: {str(e)}"
            )
            
    except HTTPException:
        raise
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
    assistant = None
    
    try:
        # Initialize the speech assistant
        assistant = SpeechAssistant()
        assistant.patient_id = patient_id
        
        # Send initial greeting
        greeting = "Welcome to Moremi AI Scheduler! Please hold while I process your conversation."
        await websocket.send_json({
            "type": "message",
            "text": greeting
        })

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
                            await websocket.send_json({
                                "type": "message",
                                "role": "system",
                                "text": status_msg
                            })
                            
                            tts_client.TTS(
                                status_msg,
                                play_locally=True
                            )
                        
                    
                    # Handle available slots
                    if assistant.suit_doc:
                        slots_response = ""
                        for doc in assistant.suit_doc:
                            slots = db_manager._get_new_date(doc['doctor_id'])
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
                    else:
                        await websocket.send_json({
                            "type": "message",
                            "text": "No doctors available at this time. Please try again later."
                        })
                        
                        tts_client.TTS(
                            "No doctors available at this time. Please try again later.",
                            play_locally=True
                        )

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
                            # Save appointment details
                            schedule_appointment(patient_id, parsed)
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
        if assistant:
            # Clean up any resources
            pass

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
        
        # Look up the reminder if a reminder_id is provided
        reminder_id = f"{type}_{patient_id}"
        reminder = None
        
        # Directly access the reminder from the dictionary using the key
        reminder = active_conversations.get(reminder_id)
        
        # If not found with simple key, try to find it in all values
        if not reminder:
            for key, value in active_conversations.items():
                if key.startswith(f"{type}_{patient_id}"):
                    reminder = value
                    break
        
        logger.info(f"Reminder lookup for {reminder_id}: {'Found' if reminder else 'Not found'}")
        
        # Initialize the appropriate agent based on type
        agent = None
        try:
            if type == "medication":
                agent = MedicalAgent(reminder or None, "medication", days_ahead=30)
            elif type == "appointment":
                agent = MedicalAgent(reminder or None, "appointment", days_ahead=30)
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Invalid conversation type: {type}",
                    "message_id": "error_invalid_type"
                })
                return
        except Exception as agent_err:
            logger.error(f"Failed to initialize agent: {str(agent_err)}")
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to initialize conversation agent: {str(agent_err)}",
                "message_id": "error_agent_init"
            })
            return
            
        # Send initial message only if we haven't sent a greeting yet
        if not conversation_state["greeting_sent"]:
            # Customize greeting based on conversation type
            if type == "medication":
                initial_message = f"Hello {reminder.patient_name if reminder else 'Patient'}! This is MinoHealth calling about your medication reminder. How are you doing today?"
            elif type == "appointment":
                initial_message = f"Hello {reminder.patient_name if reminder else 'Patient'}! This is MinoHealth calling about your upcoming appointment. How are you doing today?"
            else:
                initial_message = f"Hello {reminder.patient_name if reminder else 'Patient'}! This is MinoHealth calling. How can I assist you today?"
                
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
                
                # Try to generate speech for the initial message using TTSClient
                audio_data, sample_rate, base64_audio, _ = tts_client.TTS(
                    initial_message, 
                    play_locally=False, 
                    return_data=True
                )
                
                if base64_audio:
                    # Store audio format information
                    conversation_state["audio_format"] = {
                        "sample_rate": sample_rate,
                        "encoding": "PCM_FLOAT"
                    }
                    
                    await websocket.send_json({
                        "type": "audio",
                        "audio": base64_audio,
                        "sample_rate": sample_rate,
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
                        
                        # Process with agent - each message gets a unique ID based on timestamp and content hash
                        if agent and hasattr(agent, 'Moremi'):
                            # Add the transcription to the agent's messages
                            if not hasattr(agent, 'messages'):
                                agent.messages = []
                            agent.messages.append(Message(user_input=transcription))
                            
                            # Get response from Moremi
                            response = agent.Moremi(agent.messages, agent.system_prompt)
                            agent.messages[-1].add_response(response)
                            
                            # Send text response
                            await websocket.send_json({
                                "type": "message",
                                "role": "assistant",  # Add role to identify assistant messages
                                "text": response
                            })
                            
                            # Generate and send audio using TTSClient
                            audio_data, sample_rate, base64_audio = tts_client.TTS(
                                response,
                                play_locally=False, 
                                return_data=True
                            )
                            
                            if base64_audio:
                                # Store for frontend use
                                agent.audio_base64 = base64_audio
                                
                                # Send to client
                                await websocket.send_json({
                                    "type": "audio",
                                    "audio": base64_audio,
                                    "sample_rate": sample_rate
                                })
                        else:
                            # Fallback response if agent isn't properly initialized
                            response = f"Thank you for your message: '{transcription}'. How else can I help with your {type}?"
                            response_id = f"fallback_{conversation_state['message_count']}_{hash(response)}"
                            
                            # Send text response
                            await websocket.send_json({
                                "type": "message",
                                "role": "assistant",  # Add role to identify assistant messages
                                "text": response
                            })
                            
                            # Try to generate speech with TTSClient
                            try:
                                audio_data, sample_rate, base64_audio = tts_client.TTS(
                                    response,
                                    play_locally=False, 
                                    return_data=True
                                )
                                
                                if base64_audio:
                                    await websocket.send_json({
                                        "type": "audio",
                                        "audio": base64_audio,
                                        "sample_rate": sample_rate
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
                            audio_data, sample_rate, base64_audio, _ = tts_client.TTS(
                                farewell,
                                play_locally=False, 
                                return_data=True
                            )
                            
                            if base64_audio:
                                await websocket.send_json({
                                    "type": "audio",
                                    "audio": base64_audio,
                                    "sample_rate": sample_rate,
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
        if websocket.client_state.CONNECTED:
            await websocket.close()

# =========== APPOINTMENT MANAGER API ENDPOINTS ============ #

@app.post("/start-scheduler", response_model=List[ReminderResponse])
async def start_scheduler(request: SchedulerRequest, background_tasks: BackgroundTasks):
    """
    Start the scheduler process for either appointments or medications
    """
    try:
        with SchedulerManager(request.days_ahead) as scheduler:
            if request.type == 'appointment':
                reminders = scheduler.process_appointment_reminders(
                    hours_ahead=request.hours_ahead,
                    days_ahead=request.days_ahead
                )
            elif request.type == 'medication':
                reminders = scheduler.process_medication_reminders(
                    days_ahead=request.days_ahead
                )
            else:
                raise HTTPException(status_code=400, detail="Invalid reminder type")

            # Convert reminders to response format
            reminder_responses = []
            
            # Store active reminders for conversations with consistent keys
            for reminder in reminders:
                # Ensure patient_id is available
                patient_id = reminder.details.get('patient_id', 1)  # Default to 1 if not found
                
                # Create consistent keys for active_conversations
                simple_key = f"{reminder.message_type}_{patient_id}"
                full_key = f"{reminder.message_type}_{patient_id}_{datetime.now().timestamp()}"
                
                # Store reminder with both keys for flexibility
                active_conversations[simple_key] = reminder
                active_conversations[full_key] = reminder
                
                # Add to response
                reminder_responses.append(
                    ReminderResponse(
                        patient_name=reminder.patient_name,
                        message_type=reminder.message_type,
                        details=reminder.details,
                        timestamp=reminder.timestamp
                    )
                )
            
            logger.info(f"Stored {len(reminders)} active conversations with keys: {list(active_conversations.keys())}")
            return reminder_responses

    except Exception as e:
        logger.error(f"Error in scheduler: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start-conversation/{reminder_id}")
async def start_conversation(reminder_id: str, background_tasks: BackgroundTasks):
    """
    Start a conversation for a specific reminder
    """
    try:
        # Get the reminder from your storage/database
        reminder = active_conversations.get(reminder_id)
        if not reminder:
            raise HTTPException(status_code=404, detail="Reminder not found")

        # Initialize MedicalAgent
        agent = MedicalAgent(reminder, reminder.message_type, days_ahead=30)
        
        # Start the conversation
        background_tasks.add_task(agent.conversation_manager)

        return ConversationResponse(
            status="started",
            message="Conversation initiated",
            conversation_id=reminder_id
        )

    except Exception as e:
        logger.error(f"Error starting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/speech-input/{conversation_id}")
async def receive_speech_input(conversation_id: str):
    """
    Receive speech input for an active conversation
    """
    try:
        # Using existing transcribe_audio method which works in the event loop
        # Pass a dummy audio sample to avoid creating a new event loop with asyncio.run()
        audio_bytes = b''  # Empty bytes as placeholder
        user_input = stt_client.transcribe_audio(audio_bytes)
        
        if not user_input or not user_input.strip():
            raise HTTPException(status_code=400, detail="No speech input detected")

        return {"status": "success", "text": user_input}

    except Exception as e:
        logger.error(f"Error processing speech input: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speech-output")
async def generate_speech_output(text: str):
    """
    Generate speech output from text
    """
    try:
        # Generate speech using the TTS client
        audio_data, sample_rate, base64_audio, _ = tts_client.TTS(text, play_locally=False)
        return {
            "status": "success", 
            "message": "Speech generated successfully",
            "audio": base64_audio,
            "sample_rate": sample_rate
        }
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/end-conversation/{conversation_id}")
async def end_conversation(conversation_id: str):
    """
    End an active conversation
    """
    try:
        # End the conversation using the conversation manager
        agent = MedicalAgent(None, None)
        agent.end_conversation(conversation_id)
        return {"status": "success", "message": "Conversation ended"}

    except Exception as e:
        logger.error(f"Error ending conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-conversation-audio")
async def get_conversation_audio(conversation_id: str):
    """
    Get the audio data for a specific conversation.
    """
    try:
        audio_data = conversation_audio_cache.get(conversation_id)
        
        if not audio_data:
            raise HTTPException(status_code=404, detail="No audio found for this conversation")
            
        return {
            "conversation_id": conversation_id,
            "audio_data": audio_data
        }
        
    except Exception as e:
        logger.error(f"Error in get_conversation_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)