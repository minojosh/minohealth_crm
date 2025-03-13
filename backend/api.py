from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query, Depends, HTTPException, status, BackgroundTasks
from fastapi.websockets import WebSocketState
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from .scheduler import (
    SpeechAssistant, Message, docs, db_manager,
    speech_client, client, schedule_appointment
)
from .services.transcription_service import TranscriptionService

# Initialize services
tts_client = TTSClient(api_url=os.getenv("XTTS_URL") or os.getenv("SPEECH_SERVICE_URL"))
transcription_service = TranscriptionService(client)

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
import yaml
import re
import traceback
import base64
import numpy as np
import asyncio
from dotenv import load_dotenv

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

app = FastAPI(
    title="Moremi AI Scheduler API",
    description="API for scheduling appointments using AI assistance",
    version="1.0.0"
)

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
            # Forward finish command to STT service
            logger.info("Finish command received, proxying to STT service")
            return {"transcription": "", "status": "finished"}
        
        if 'audio' in data:
            # Process audio data using transcription service
            transcription = await transcription_service.transcribe(data['audio'])
            
            if transcription:
                logger.info(f"Transcription result: {transcription}")
                return {"transcription": transcription}
            else:
                return {"transcription": "No speech detected"}
        
        return {"error": "Invalid request format"}
    
    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {str(e)}")
        return {"error": f"Internal server error: {str(e)}"}

@app.post("/api/extract/audio")
async def extract_from_audio(request: Request):
    """Extract structured data directly from audio using the medical Assistant"""
    try:
        data = await request.json()
        audio_base64 = data.get("audio")
        
        if not audio_base64:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        # Transcribe audio using our service
        transcript = await transcription_service.transcribe(audio_base64)
        
        if not transcript:
            raise HTTPException(status_code=400, detail="No speech detected in audio")
                
        logger.info(f"Transcription result: {transcript}")
        
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
    """
    WebSocket endpoint for handling audio streaming and transcription.
    Now supports both streaming mode and complete recording processing.
    """
    # Check authentication token
    if token not in API_KEYS.values():
        logger.warning(f"Authentication failed for WebSocket connection: {token}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
        
    await websocket.accept()
    logger.info("Audio WebSocket connection established")
    
    try:
        await websocket.send_json({
            "type": "connection_status",
            "status": "connected", 
            "message": "WebSocket connection established successfully"
        })
        
        while True:
            # Receive message from client
            try:
                data = await websocket.receive_json()
                message_type = data.get("type")
                payload = data.get("payload", {})
                timestamp = data.get("timestamp", 0)
                
                if message_type == "transcription":
                    try:
                        # Process audio data received from client
                        audio_data = payload.get("audioData")
                        sample_rate = payload.get("sampleRate", 16000)
                        process_complete = payload.get("processComplete", False)
                        
                        if not audio_data:
                            raise ValueError("No audio data provided")
                        
                        # Use transcription service for consistent handling
                        transcription = await transcription_service.transcribe(audio_data)
                        
                        if not transcription:
                            logger.warning("No speech detected in audio")
                            await websocket.send_json({
                                "type": "transcription",
                                "payload": {
                                    "text": "No speech detected",
                                    "transcription": "No speech detected", 
                                    "confidence": 0.0,
                                    "isComplete": process_complete
                                },
                                "timestamp": timestamp
                            })
                        else:
                            # Valid transcription
                            logger.info(f"Valid transcription received: '{transcription}'")
                            await websocket.send_json({
                                "type": "transcription",
                                "payload": {
                                    "text": transcription,
                                    "transcription": transcription,
                                    "confidence": 0.9,
                                    "isComplete": process_complete
                                },
                                "timestamp": timestamp
                            })
                        
                    except Exception as e:
                        logger.error(f"Error processing audio data: {str(e)}")
                        await websocket.send_json({
                            "type": "error",
                            "payload": {
                                "message": f"Error processing audio: {str(e)}"
                            },
                            "timestamp": timestamp
                        })
                elif message_type == "ping":
                    # Add a ping handler to keep connection alive
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": timestamp
                    })
                else:
                    # Handle other message types
                    await websocket.send_json({
                        "type": "status",
                        "payload": {
                            "message": f"Received message type: {message_type}"
                        },
                        "timestamp": timestamp
                    })
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                await websocket.send_json({
                    "type": "error",
                    "payload": {
                        "message": "Invalid JSON format"
                    }
                })
                continue
            except asyncio.CancelledError:
                logger.info("WebSocket task was cancelled")
                break
                    
    except WebSocketDisconnect:
        logger.info("Audio WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Error in audio websocket: {str(e)}")
    finally:
        if websocket.client_state == WebSocket.client_state.CONNECTED:
            await websocket.close()
        logger.info("Audio WebSocket connection closed")

@app.websocket("/ws/schedule/{patient_id}")
async def schedule_session(websocket: WebSocket, patient_id: int):
    """
    WebSocket endpoint that handles the entire scheduling flow
    """
    await websocket.accept()
    assistant = None
    
    try:
        # Initialize the speech assistant
        assistant = SpeechAssistant()
        assistant.patient_id = patient_id
        
        # Send initial greeting
        greeting = "Welcome to Moremi AI Scheduler! Please send the conversation context to begin."
        await websocket.send_json({
            "type": "message",
            "text": greeting
        })

        # Wait for context from client
        try:
            message = await websocket.receive_json()
            context = message.get("context")
            if not context:
                raise ValueError("No context provided")
                
            # Process initial context and specialty search
            initial_question = assistant.manage_context(context)  # Pass the received context
            assistant.messages.append(Message(user_input=initial_question))
            
            # Send acknowledgment
            await websocket.send_json({
                "type": "message",
                "text": "Context received. I am listening..."
            })
            
            # Generate and send TTS
            audio_data, sample_rate = speech_client.generate_speech("Context received. I am listening...")
            if audio_data is not None:
                base64_audio = convert_audio_to_base64(audio_data)
                await websocket.send_json({
                    "type": "audio",
                    "audio": base64_audio,
                    "sample_rate": sample_rate
                })

            # Get initial Moremi response for specialty/doctor matching
            response = assistant.get_moremi_response(assistant.messages, assistant.system_prompt)
            assistant.messages[-1].add_response(str(response))
            
            # Process specialty/doctor search
            if 'search_specialty' in str(response).lower():
                text = "Finding a suitable doctor"
                await websocket.send_json({
                    "type": "message",
                    "text": text
                })
                
                # Generate and send TTS
                audio_data, sample_rate = speech_client.generate_speech(text)
                if audio_data is not None:
                    base64_audio = convert_audio_to_base64(audio_data)
                    await websocket.send_json({
                        "type": "audio",
                        "audio": base64_audio,
                        "sample_rate": sample_rate
                    })
                
                for doc in docs:
                    specialty = doc['specialty']
                    if specialty.lower() in response.lower():
                        assistant.suit_doc.append(doc)
                        
                if len(assistant.suit_doc) == 0:
                    msg = "The required specialty is not available in our facility. Please speak to the present doctor about a referral."
                    await websocket.send_json({
                        "type": "message",
                        "text": msg
                    })
                    audio_data, sample_rate = speech_client.generate_speech(msg)
                    if audio_data is not None:
                        base64_audio = convert_audio_to_base64(audio_data)
                        await websocket.send_json({
                            "type": "audio",
                            "audio": base64_audio,
                            "sample_rate": sample_rate
                        })
                
            else:
                text = 'Finding available days for the doctor'
                await websocket.send_json({
                    "type": "message",
                    "text": text
                })
                
                # Generate and send TTS
                audio_data, sample_rate = speech_client.generate_speech(text)
                if audio_data is not None:
                    base64_audio = convert_audio_to_base64(audio_data)
                    await websocket.send_json({
                        "type": "audio",
                        "audio": base64_audio,
                        "sample_rate": sample_rate
                    })
                
                for doc in docs:
                    if doc['doctor_name'] in response:
                        assistant.suit_doc.append(doc)
                        
                if len(assistant.suit_doc) == 0:
                    msg = "The required doctor is not present in our facility. Please speak to the present doctor about a referral."
                    await websocket.send_json({
                        "type": "message",
                        "text": msg
                    })
                    audio_data, sample_rate = speech_client.generate_speech(msg)
                    if audio_data is not None:
                        base64_audio = convert_audio_to_base64(audio_data)
                        await websocket.send_json({
                            "type": "audio",
                            "audio": base64_audio,
                            "sample_rate": sample_rate
                        })

            # If doctors were found, process and send available slots
            if len(assistant.suit_doc) != 0:
                doctors_with_slots = []
                for doc in assistant.suit_doc:
                    slots = []
                    while True:
                        new_slots = db_manager._get_new_date(doc['doctor_id'])
                        if len(new_slots) != 0 or db_manager.days_ahead >= 90:
                            slots.extend([slot.strftime('%Y-%m-%d %H:%M:%S') 
                                        for slot in new_slots])
                            break
                        db_manager.days_ahead += 5
                    
                    doc_info = doc.copy()
                    doc_info['available_slots'] = slots
                    doctors_with_slots.append(doc_info)

                await websocket.send_json({
                    "type": "doctors",
                    "data": doctors_with_slots
                })

            # Continue with the existing command loop
            while True:
                try:
                    message = await websocket.receive_json()
                    command = message.get("command")

                    if command == "start_listening":
                        # Start speech recognition
                        transcription = client.recognize_speech()
                        
                        if transcription.strip():
                            # Send transcription to client
                            await websocket.send_json({
                                "type": "transcription",
                                "text": transcription
                            })

                            # Process with Moremi
                            assistant.messages.append(Message(user_input=transcription))
                            response = assistant.get_moremi_response(
                                assistant.messages,
                                assistant.system_prompt
                            )
                            assistant.messages[-1].add_response(response)

                            # Send response text
                            await websocket.send_json({
                                "type": "message",
                                "text": response
                            })

                            # Generate and send TTS
                            audio_data, sample_rate, base64_audio, _ = tts_client.TTS(
                                response, 
                                play_locally=False, 
                                return_data=True
                            )
                            
                            if base64_audio:
                                # Store for frontend use
                                assistant.audio_base64 = base64_audio
                                
                                # Cache for later retrieval
                                conversation_audio_cache[f"schedule_{patient_id}"] = {
                                    "audio": base64_audio,
                                    "sample_rate": sample_rate,
                                    "encoding": "PCM_FLOAT"
                                }
                                
                                # Send to client
                                await websocket.send_json({
                                    "type": "audio",
                                    "audio": base64_audio,
                                    "sample_rate": sample_rate,
                                    "message_id": f"audio_schedule_{patient_id}",
                                    "encoding": "PCM_FLOAT"
                                })

                            # If doctors were found, send them
                            if assistant.suit_doc:
                                # Process available slots
                                doctors_with_slots = []
                                for doc in assistant.suit_doc:
                                    slots = []
                                    while True:
                                        new_slots = db_manager._get_new_date(doc['doctor_id'])
                                        if len(new_slots) != 0 or db_manager.days_ahead >= 90:
                                            slots.extend([slot.strftime('%Y-%m-%d %H:%M:%S') 
                                                        for slot in new_slots])
                                            break
                                        db_manager.days_ahead += 5
                                    
                                    doc_info = doc.copy()
                                    doc_info['available_slots'] = slots
                                    doctors_with_slots.append(doc_info)

                                await websocket.send_json({
                                    "type": "doctors",
                                    "data": doctors_with_slots
                                })

                    elif command == "finish":
                        # Get conversation summary
                        assistant.messages.append(
                            Message(user_input="Extract the agreed upon schedule from the interaction history. If there was no agreed upon appointment respond with the word null.")
                        )
                        summary = assistant.get_moremi_response(
                            assistant.messages,
                            assistant.summary_system_prompt
                        )

                        # Send summary to client
                        await websocket.send_json({
                            "type": "summary",
                            "text": summary
                        })

                        # Generate and send TTS for summary
                        audio_data, sample_rate, base64_audio, _ = tts_client.TTS(
                            f"Summary: {summary}", 
                            play_locally=False, 
                            return_data=True
                        )
                        
                        if base64_audio:
                            # Store for frontend use
                            assistant.audio_base64 = base64_audio
                            
                            # Cache for later retrieval
                            conversation_audio_cache[f"summary_{patient_id}"] = {
                                "audio": base64_audio,
                                "sample_rate": sample_rate,
                                "encoding": "PCM_FLOAT"
                            }
                            
                            # Send to client
                            await websocket.send_json({
                                "type": "audio",
                                "audio": base64_audio,
                                "sample_rate": sample_rate,
                                "message_id": f"audio_summary_{patient_id}",
                                "encoding": "PCM_FLOAT"
                            })

                        if summary != "null":
                            try:
                                # Parse and save appointment
                                parsed_dict = json.loads(summary)
                                schedule_appointment(patient_id, parsed_dict)
                                
                                await websocket.send_json({
                                    "type": "appointment",
                                    "status": "success",
                                    "data": parsed_dict
                                })

                                # Generate and send TTS for confirmation
                                confirm_message = f"Your appointment with {parsed_dict['doctorName']} has been scheduled for {parsed_dict['appointmentDateTime']}."
                                audio_data, sample_rate, base64_audio, _ = tts_client.TTS(
                                    confirm_message, 
                                    play_locally=False, 
                                    return_data=True
                                )
                                
                                if base64_audio:
                                    await websocket.send_json({
                                        "type": "audio",
                                        "audio": base64_audio,
                                        "sample_rate": sample_rate,
                                        "message_id": f"audio_confirm_{patient_id}",
                                        "encoding": "PCM_FLOAT"
                                    })
                            except Exception as e:
                                logger.error(f"Error saving appointment: {str(e)}")
                                error_message = "Failed to save appointment"
                                await websocket.send_json({
                                    "type": "error",
                                    "message": error_message
                                })
                                
                                # Generate and send TTS for error
                                audio_data, sample_rate, base64_audio, _ = tts_client.TTS(
                                    error_message, 
                                    play_locally=False, 
                                    return_data=True
                                )
                                
                                if base64_audio:
                                    await websocket.send_json({
                                        "type": "audio",
                                        "audio": base64_audio,
                                        "sample_rate": sample_rate,
                                        "message_id": f"audio_error_{patient_id}",
                                        "encoding": "PCM_FLOAT"
                                    })
                        
                        break

                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    continue

        except json.JSONDecodeError:
            await websocket.send_json({
                "type": "error",
                "message": "Invalid JSON format for context"
            })
            return
        except ValueError as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
            return

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in scheduling session: {str(e)}")
        if websocket.client_state.CONNECTED:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    finally:
        if assistant:
            assistant.save_conversation()
        await websocket.close()

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
                        transcription = client.transcribe_audio(audio_bytes)
                        
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
                            "type": "transcription",
                            "text": transcription,
                            "message_id": transcription_id
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
                            
                            # Generate a unique response ID
                            response_id = f"resp_{conversation_state['message_count']}_{hash(response)}"
                            
                            # Only send if this is a new response (prevent duplicates)
                            if response_id != conversation_state["last_message_id"]:
                                # Send text response
                                await websocket.send_json({
                                    "type": "message",
                                    "text": response,
                                    "message_id": response_id
                                })
                                
                                conversation_state["message_count"] += 1
                                conversation_state["last_message"] = response
                                conversation_state["last_message_id"] = response_id
                                
                                # Generate and send audio using TTSClient
                                try:
                                    audio_data, sample_rate, base64_audio, _ = tts_client.TTS(
                                        response,
                                        play_locally=False, 
                                        return_data=True
                                    )
                                    
                                    if base64_audio:
                                        # Store for frontend use
                                        agent.audio_base64 = base64_audio
                                        
                                        # Cache for later retrieval
                                        conversation_audio_cache[conversation_id or reminder_id] = {
                                            "audio": base64_audio,
                                            "sample_rate": sample_rate,
                                            "encoding": "PCM_FLOAT"
                                        }
                                        
                                        # Send to client
                                        await websocket.send_json({
                                            "type": "audio",
                                            "audio": base64_audio,
                                            "sample_rate": sample_rate,
                                            "message_id": f"audio_{response_id}",
                                            "encoding": "PCM_FLOAT"
                                        })
                                        
                                        # Send a prompt to encourage continued conversation
                                        await websocket.send_json({
                                            "type": "status",
                                            "status": "waiting_for_input",
                                            "message": "Please speak now",
                                            "message_id": f"prompt_after_{response_id}"
                                        })
                                except Exception as e:
                                    logger.error(f"Failed to generate speech for response: {str(e)}")
                        else:
                            # Fallback response if agent isn't properly initialized
                            response = f"Thank you for your message: '{transcription}'. How else can I help with your {type}?"
                            response_id = f"fallback_{conversation_state['message_count']}_{hash(response)}"
                            
                            # Send text response only if it's not a duplicate
                            if response_id != conversation_state["last_message_id"]:
                                await websocket.send_json({
                                    "type": "message",
                                    "text": response,
                                    "message_id": response_id
                                })
                                
                                conversation_state["message_count"] += 1
                                conversation_state["last_message"] = response
                                conversation_state["last_message_id"] = response_id
                                
                                # Try to generate speech with TTSClient
                                try:
                                    audio_data, sample_rate, base64_audio, _ = tts_client.TTS(
                                        response,
                                        play_locally=False, 
                                        return_data=True
                                    )
                                    
                                    if base64_audio:
                                        await websocket.send_json({
                                            "type": "audio",
                                            "audio": base64_audio,
                                            "sample_rate": sample_rate,
                                            "message_id": f"audio_{response_id}",
                                            "encoding": "PCM_FLOAT"
                                        })
                                        
                                        # Send a prompt to encourage continued conversation
                                        await websocket.send_json({
                                            "type": "status",
                                            "status": "waiting_for_input",
                                            "message": "Please speak now",
                                            "message_id": f"prompt_after_{response_id}"
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
        # Save conversation if agent was initialized
        if 'agent' in locals() and agent:
            try:
                # Save conversation history
                if hasattr(agent, 'save_conversation'):
                    agent.save_conversation()
            except Exception as save_err:
                logger.error(f"Error saving conversation: {str(save_err)}")
        
        # Ensure WebSocket is closed
        if 'websocket' in locals() and websocket.client_state != WebSocketState.DISCONNECTED:
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
        user_input = client.transcribe_audio(audio_bytes)
        
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

@app.get("/test-moremi-connection")
async def test_moremi_connection():
    """Test the connection to the Moremi API."""
    try:
        from .moremi import ConversationManager
        
        base_url = os.getenv("MOREMI_API_BASE_URL", "http://localhost:8000")
        api_key = os.getenv("MOREMI_API_KEY") or os.getenv("OPENAI_API_KEY")
        logger.info(f"Testing connection to Moremi API at: {base_url}")
        
        # Validate environment variables
        env_status = {
            "MOREMI_API_BASE_URL": "✅ Set" if base_url != "http://localhost:8000" else "⚠️ Using default",
            "MOREMI_API_KEY": "✅ Set" if api_key else "❌ Missing"
        }
        
        logger.info(f"Environment variables: {env_status}")
        
        try:
            if not api_key:
                return {
                    "status": "error",
                    "message": "API key not found. Set MOREMI_API_KEY or OPENAI_API_KEY environment variable",
                    "environment": env_status
                }
                
            conversation = ConversationManager(base_url, api_key)
            conversation.add_user_message(text="Hello")
            response = conversation.get_assistant_response(stream=False)
            
            return {
                "status": "success",
                "message": "Successfully connected to Moremi API",
                "response_preview": response[:100] + "..." if response else "No response",
                "environment": env_status
            }
        except Exception as e:
            logger.error(f"Connection test failed with error: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "environment": env_status
            }
            
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to Moremi API: {str(e)}"
        )

@app.get("/api/patient/{patient_id}")
async def get_patient(patient_id: int):
    """Get patient details by ID"""
    try:
        # Import database manager
        from .database_utils import PatientDatabaseManager
        
        # Create database manager
        db_manager = PatientDatabaseManager()
        
        # Get patient details
        patient = db_manager.get_patient(patient_id)
        
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient with ID {patient_id} not found")
        
        # Get patient's appointments
        appointments = db_manager.get_patient_appointments(patient_id)
        
        # Get patient's medical conditions
        medical_conditions = db_manager.get_patient_medical_conditions(patient_id)
        
        # Return patient details
        return {
            "status": "success",
            "patient": patient,
            "appointments": appointments,
            "medical_conditions": medical_conditions
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting patient details: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Server error getting patient details: {str(e)}"
        )

@app.get("/api/patients")
async def get_patients():
    """Get all patients"""
    try:
        # Import database manager
        from .database_utils import PatientDatabaseManager
        
        # Create database manager
        db_manager = PatientDatabaseManager()
        
        # Get all patients
        patients = db_manager.get_all_patients()
        
        # Return patients
        return {
            "status": "success",
            "patients": patients
        }
    except Exception as e:
        logger.error(f"Error getting patients: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Server error getting patients: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)