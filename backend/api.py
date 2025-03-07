from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query, Depends, HTTPException, status, BackgroundTasks
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging
from .scheduler import (
    SpeechAssistant, Message, docs, db_manager,
    speech_client, client, schedule_appointment
)
# Import appointment manager components
from .appointment_manager import (
    SchedulerManager, 
    MedicalAgent, 
    ReminderMessage
)
from datetime import datetime
import json
import asyncio
import fastapi # Import the fastapi module itself
import base64
import numpy as np
from typing import Optional, List
from pydantic import BaseModel
from .TTS_client import TTSClient

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

app = FastAPI(
    title="Moremi AI Scheduler API",
    description="API for scheduling appointments using AI assistance",
    version="1.0.0"
)

# Initialize TTS client
tts_client = TTSClient()

print(f"FastAPI version: {fastapi.__version__}") # Print version at startup

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        
        # Use the TTS client to generate audio
        audio_data, sample_rate, base64_audio = tts_client.TTS(
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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate speech: {str(e)}"
        )

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
            # Process audio data
            audio_data = data['audio']
            try:
                # Convert from base64 to bytes
                audio_bytes = base64.b64decode(audio_data)
                
                # Process using the STT client
                transcription = client.transcribe_audio(audio_bytes)
                
                logger.info(f"Transcription result: {transcription}")
                return {"transcription": transcription}
            except Exception as e:
                logger.error(f"Error processing audio data: {str(e)}")
                return {"error": f"Failed to process audio: {str(e)}"}
        
        return {"error": "Invalid request format"}
    
    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {str(e)}")
        return {"error": f"Internal server error: {str(e)}"}

@app.post("/api/extract/data")
async def extract_data(request: Request):
    """
    Extract structured data from a transcript using Moremi.
    """
    try:
        data = await request.json()
        transcript = data.get("transcript")
        
        if not transcript:
            raise HTTPException(
                status_code=400,
                detail="No transcript provided"
            )
        
        # Initialize the speech assistant
        assistant = SpeechAssistant()
        assistant.messages.append(Message(user_input=transcript))
        
        # Get Moremi response
        response = assistant.get_moremi_response(assistant.messages, assistant.system_prompt)
        
        # Save the results (this will process and clean the YAML)
        timestamp = assistant._get_timestamp()
        raw_path, processed_path = assistant._save_yaml(response, timestamp)
        
        # Return the processed YAML data
        import yaml
        with open(processed_path, 'r') as f:
            extracted_data = yaml.safe_load(f)
            
        return extracted_data
        
    except Exception as e:
        logger.error(f"Error in data extraction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract data: {str(e)}"
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
                        
                        # Decode base64 audio data
                        audio_bytes = base64.b64decode(audio_data)
                        
                        # Convert to float32 numpy array
                        audio_np = safe_frombuffer(audio_bytes, dtype=np.float32)
                        
                        # Log the audio processing approach
                        if process_complete:
                            logger.info(f"Processing complete audio recording of {len(audio_np)/sample_rate:.2f} seconds")
                        else:
                            logger.info(f"Processing streaming audio chunk of {len(audio_np)/sample_rate:.2f} seconds")
                        
                        # Use the client's transcribe_audio method
                        transcription = client.transcribe_audio(audio_np.tobytes())
                        logger.info(f"Transcription result: {transcription}")
                        
                        # Proper handling of different response types:
                        # 1. Only treat as "No speech detected" if the transcription is specifically that phrase
                        # 2. Handle error messages as errors
                        # 3. Any other response is considered a valid transcription
                        if not transcription:
                            logger.warning("Empty transcription received from STT service")
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
                        elif transcription == "No speech detected" or transcription == "No transcription result received":
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
                        elif transcription.startswith(("Error", "Connection error", "Server error")):
                            # This is an actual error, not a transcription
                            logger.error(f"Error from STT service: {transcription}")
                            await websocket.send_json({
                                "type": "error",
                                "payload": {
                                    "message": transcription
                                },
                                "timestamp": timestamp
                            })
                        else:
                            # Any other response is considered a valid transcription
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
                            audio_data, sample_rate = speech_client.generate_speech(response)
                            if audio_data is not None:
                                base64_audio = convert_audio_to_base64(audio_data)
                                await websocket.send_json({
                                    "type": "audio",
                                    "audio": base64_audio,
                                    "sample_rate": sample_rate
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
                        audio_data, sample_rate = speech_client.generate_speech(f"Summary: {summary}")
                        if audio_data is not None:
                            base64_audio = convert_audio_to_base64(audio_data)
                            await websocket.send_json({
                                "type": "audio",
                                "audio": base64_audio,
                                "sample_rate": sample_rate
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
                                audio_data, sample_rate = speech_client.generate_speech(confirm_message)
                                if audio_data is not None:
                                    base64_audio = convert_audio_to_base64(audio_data)
                                    await websocket.send_json({
                                        "type": "audio",
                                        "audio": base64_audio,
                                        "sample_rate": sample_rate
                                    })
                            except Exception as e:
                                logger.error(f"Error saving appointment: {str(e)}")
                                error_message = "Failed to save appointment"
                                await websocket.send_json({
                                    "type": "error",
                                    "message": error_message
                                })
                                
                                # Generate and send TTS for error
                                audio_data, sample_rate = speech_client.generate_speech(error_message)
                                if audio_data is not None:
                                    base64_audio = convert_audio_to_base64(audio_data)
                                    await websocket.send_json({
                                        "type": "audio",
                                        "audio": base64_audio,
                                        "sample_rate": sample_rate
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
        
    await websocket.accept()
    logger.info(f"Conversation WebSocket connection established for patient {patient_id} with type {type}")
    
    try:
        # Send connection successful message
        await websocket.send_json({
            "type": "connection_status",
            "status": "connected",
            "message": "WebSocket connection established successfully"
        })
        
        # Create reminder ID based on type and patient ID
        reminder_id = f"{type}_{patient_id}"
        reminder = active_conversations.get(reminder_id)
        
        if not reminder:
            logger.warning(f"No active reminder found for {reminder_id}")
            logger.info(f"Available reminders: {list(active_conversations.keys())}")
            
            # Create a dummy reminder for debugging purposes
            logger.info("Creating a dummy reminder for testing")
            dummy_greeting = f"This is a test {type} reminder. No actual reminder data was found for patient {patient_id}."
            
            await websocket.send_json({
                "type": "message",
                "text": dummy_greeting
            })
            
            # Try to generate audio for the dummy greeting
            try:
                audio_data, sample_rate, base64_audio = tts_client.TTS(
                    dummy_greeting, 
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
                logger.error(f"Failed to generate speech for dummy greeting: {str(audio_err)}")
            
            return
            
        logger.info(f"Found reminder for {reminder_id}: {reminder.message_type} for {reminder.patient_name}")
        
        # Initialize the appropriate agent based on type
        agent = None
        try:
            if type == "medication":
                agent = MedicalAgent(reminder, "medication", days_ahead=30)  # Add days_ahead parameter
            elif type == "appointment":
                agent = MedicalAgent(reminder, "appointment", days_ahead=30)  # Add days_ahead parameter
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Invalid conversation type: {type}"
                })
                return
        except Exception as agent_err:
            logger.error(f"Failed to initialize agent: {str(agent_err)}")
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to initialize conversation agent: {str(agent_err)}"
            })
            return
            
        # Send initial message
        initial_message = f"Starting {type} conversation for patient {patient_id}. Hello {reminder.patient_name}!"
        await websocket.send_json({
            "type": "message",
            "text": initial_message
        })
        
        # Try to generate speech for the initial message using TTSClient
        try:
            audio_data, sample_rate, base64_audio = tts_client.TTS(
                initial_message, 
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
                    
                    if not audio_base64:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No audio data provided"
                        })
                        continue
                    
                    # Process audio for transcription
                    try:
                        audio_bytes = base64.b64decode(audio_base64)
                        transcription = client.transcribe_audio(audio_bytes)
                        
                        if not transcription or transcription == "No speech detected":
                            await websocket.send_json({
                                "type": "message",
                                "text": "I couldn't hear you clearly. Could you please repeat that?"
                            })
                            continue
                            
                        # Send transcription back to client
                        await websocket.send_json({
                            "type": "transcription",
                            "text": transcription
                        })
                        
                        # Process with agent
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
                            
                            # Send text response
                            await websocket.send_json({
                                "type": "message",
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
                            "message": "Failed to process your audio"
                        })
                
                # Handle ping to keep connection alive
                elif message_type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().timestamp()
                    })
                
                # Handle commands
                elif message_type == "command":
                    command = data.get("command")
                    
                    if command == "finish":
                        farewell = "Thank you for using our service. Goodbye!"
                        await websocket.send_json({
                            "type": "message",
                            "text": farewell
                        })
                        
                        # Generate farewell audio
                        try:
                            audio_data, sample_rate, base64_audio = tts_client.TTS(
                                farewell,
                                play_locally=False, 
                                return_data=True
                            )
                            
                            if base64_audio:
                                await websocket.send_json({
                                    "type": "audio",
                                    "audio": base64_audio,
                                    "sample_rate": sample_rate
                                })
                        except Exception as e:
                            logger.error(f"Error generating farewell audio: {e}")
                            
                        break
                    else:
                        await websocket.send_json({
                            "type": "status",
                            "message": f"Received command: {command}"
                        })
                
                # Handle other message types
                else:
                    await websocket.send_json({
                        "type": "status",
                        "message": f"Received unknown message type: {message_type}"
                    })
                    
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from WebSocket message")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format received"
                })
                continue
            except WebSocketDisconnect:
                logger.info(f"Client disconnected from {type} conversation for patient {patient_id}")
                break
            except asyncio.CancelledError:
                logger.info("WebSocket task was cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Internal server error"
                    })
                except:
                    break
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from {type} conversation for patient {patient_id}")
    except Exception as e:
        logger.error(f"Error in conversation websocket: {str(e)}")
    finally:
        if websocket.client_state == WebSocket.client_state.CONNECTED:
            await websocket.close()
        logger.info(f"Conversation WebSocket connection closed for patient {patient_id}")

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

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)