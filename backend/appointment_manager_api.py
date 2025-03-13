from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from .appointment_manager import (
    SchedulerManager, 
    MedicalAgent, 
    ReminderMessage,
    SpeechRecognitionClient
)
from .XTTS_adapter import TTSClient
from datetime import datetime
import logging
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Appointment Scheduler API")

# Initialize clients
speech_client = SpeechRecognitionClient()
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path, encoding='utf-8')
tts_client = TTSClient(api_url=os.getenv("XTTS_URL") or os.getenv("SPEECH_SERVICE_URL"))

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

# Store active conversations
active_conversations = {}
# Store last audio data for each conversation
conversation_audio_cache = {}

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
            reminder_responses = [
                ReminderResponse(
                    patient_name=reminder.patient_name,
                    message_type=reminder.message_type,
                    details=reminder.details,
                    timestamp=reminder.timestamp
                )
                for reminder in reminders
            ]

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
        agent = MedicalAgent(reminder, reminder.message_type, days_ahead=30)  # Configure days_ahead as needed
        
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
        # Get speech input using the speech recognition client
        user_input = speech_client.recognize_speech()
        
        if not user_input.strip():
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
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
