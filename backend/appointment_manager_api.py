from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
from .appointment_manager import (
    SchedulerManager, 
    MedicalAgent, 
    ReminderMessage,
    SpeechRecognitionClient,
    VoiceAssistant,
    schedule_appointment,
    Appointment,
    Patient, 
    Doctor
)
from .XTTS_adapter import TTSClient
from datetime import datetime
import logging
import json
from dotenv import load_dotenv
import os
import base64
import uuid
import asyncio
from pathlib import Path

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
speech_client = SpeechRecognitionClient()
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path, encoding='utf-8')
tts_client = TTSClient(api_url=os.getenv("XTTS_URL") or os.getenv("SPEECH_SERVICE_URL"))

# Store active conversations, agents, and reminders
active_conversations: Dict[str, MedicalAgent] = {}
active_voice_assistants: Dict[str, VoiceAssistant] = {}
active_reminders: Dict[str, ReminderMessage] = {}

# Create a centralized service for managing reminders and conversations
class ReminderService:
    @staticmethod
    def create_reminder(patient_name: str, message_type: str, details: Dict[str, Any]) -> str:
        """Create a new reminder and return its ID"""
        reminder_id = str(uuid.uuid4())
        
        reminder = ReminderMessage(
            patient_name=patient_name,
            message_type=message_type,
            details=details
        )
        
        active_reminders[reminder_id] = reminder
        return reminder_id
    
    @staticmethod
    def get_reminder(reminder_id: str) -> Optional[ReminderMessage]:
        """Get a reminder by its ID"""
        return active_reminders.get(reminder_id)
    
    @staticmethod
    def create_or_get_agent(reminder_id: str, days_ahead: int = 30) -> MedicalAgent:
        """Create or get an existing agent for a reminder"""
        if reminder_id in active_conversations:
            return active_conversations[reminder_id]
        
        reminder = ReminderService.get_reminder(reminder_id)
        if not reminder:
            raise ValueError(f"No reminder found with ID: {reminder_id}")
        
        agent = MedicalAgent(
            reminder=reminder,
            message_type=reminder.message_type,
            days_ahead=days_ahead
        )
        
        # Set the appropriate system prompt
        if agent.message_type == 'appointment':
            agent.system_prompt = agent.appointment_prompt
        else:  # medication
            agent.system_prompt = agent.medication_prompt.format(
                medication_dosage=agent.reminder.details.get('dosage', 'as prescribed'),
                medication_frequency=agent.reminder.details.get('frequency', 'as directed')
            )
        
        active_conversations[reminder_id] = agent
        return agent
    
    @staticmethod
    def create_or_get_voice_assistant(reminder_id: str) -> VoiceAssistant:
        """Create or get an existing voice assistant for a reminder"""
        if reminder_id in active_voice_assistants:
            return active_voice_assistants[reminder_id]
        
        voice_assistant = VoiceAssistant()
        active_voice_assistants[reminder_id] = voice_assistant
        return voice_assistant
    
    @staticmethod
    def generate_initial_message(reminder_id: str) -> str:
        """Generate an initial message for a conversation based on reminder type"""
        agent = ReminderService.create_or_get_agent(reminder_id)
        reminder = ReminderService.get_reminder(reminder_id)
        
        if not reminder:
            raise ValueError(f"No reminder found with ID: {reminder_id}")
        
        if agent.message_type == 'appointment':
            doctor_name = reminder.details.get('doctor_name', 'your doctor')
            appointment_datetime = reminder.details.get('datetime', datetime.now())
            
            if isinstance(appointment_datetime, str):
                try:
                    appointment_datetime = datetime.fromisoformat(appointment_datetime)
                    appointment_datetime = appointment_datetime.strftime('%B %d at %I:%M %p')
                except:
                    # Keep as is if parsing fails
                    pass
                
            dt = appointment_datetime
            
            # Format to natural representation
            appointment_datetime = dt.strftime("%d{} %B at %I:%M %p").format(
                "th" if 4 <= dt.day <= 20 or 24 <= dt.day <= 30
                else {1: "st", 2: "nd", 3: "rd"}.get(dt.day % 10, "th")
            )
                        
            initial_message = agent.initial_appointment_message.format(
                patient_name=reminder.patient_name,
                doctor_name=doctor_name,
                appointment_datetime=appointment_datetime
            )
        else:  # medication
            initial_message = agent.initial_medication_message.format(
                patient_name=reminder.patient_name,
                medication_name=reminder.details.get('medication_name', 'your medication'),
                medication_dosage=reminder.details.get('dosage', 'as prescribed'),
                medication_frequency=reminder.details.get('frequency', 'as directed')
            )
        
        # Initialize conversation history
        agent.LLM.conversation_history.append({
            "role": "user",
            "content": "Hi"
        })
        agent.LLM.conversation_history.append({
            "role": "assistant",
            "content": initial_message
        })
        
        return initial_message

    @staticmethod
    def end_conversation(reminder_id: str, days_ahead: int = 30) -> Dict[str, Any]:
        """End a conversation and process the results"""
        agent = ReminderService.create_or_get_agent(reminder_id)
        
        if agent.message_type == 'appointment':
            try:
                # Extract appointment details from conversation
                context = agent.format_conversation(agent.LLM.conversation_history)
                final = agent.moremi_response(context, agent.extract_prompt)
                new_datetime = agent.extract_datetime_from_moremi(final)
                # Normalize new_datetime format
                new_datetime_obj = datetime.strptime(new_datetime, '%Y-%m-%d %H:%M')
                
                reminder = ReminderService.get_reminder(reminder_id)
        
                if not reminder:
                    raise ValueError(f"No reminder found with ID: {reminder_id}")
    
                old_datetime = reminder.details.get('datetime')
                
                # Normalize datetime format
                if isinstance(old_datetime, str):
                    old_datetime_obj = datetime.strptime(old_datetime, '%Y-%m-%d %H:%M')
                else:
                    # Convert existing datetime to our format
                    temp_str = old_datetime.strftime('%Y-%m-%d %H:%M')
                    old_datetime_obj = datetime.strptime(temp_str, '%Y-%m-%d %H:%M')
                
                
                        
                # Get doctor and patient IDs from the reminder
                doctor_id = agent.reminder.details.get('doctor_id')
                patient_id = agent.reminder.details.get('patient_id')
                
                if old_datetime_obj == new_datetime_obj:
                    logger.info("Patient Confirmed the existing appointment")
                    with SchedulerManager(days_ahead) as scheduler:
                        # Find the appointment for the patient
                        appointment = scheduler.session.query(Appointment).filter_by(
                            patient_id=patient_id,
                            doctor_id=doctor_id
                        ).first()
                        # Check if patient exists
                        patient = scheduler.session.query(Patient).filter_by(patient_id=patient_id).first()
                        # Get doctor details for the response
                        doctor = scheduler.session.query(Doctor).filter_by(doctor_id=doctor_id).first()
                        
                        # Update the appointment datetime
                        appointment.datetime = new_datetime_obj
                else:   
                    logger.info("Patient asked for a reschedule")
                    # Schedule the appointment using the appointment manager
                    with SchedulerManager(days_ahead) as scheduler:
                        # Check if patient exists
                        patient = scheduler.session.query(Patient).filter_by(patient_id=patient_id).first()
                        # Get doctor details for the response
                        doctor = scheduler.session.query(Doctor).filter_by(doctor_id=doctor_id).first()
                        if not patient:
                            return {
                                "success": False,
                                "message": f"Patient with ID {patient_id} not found"
                            }
                        
                        # Find the appointment for the patient
                        appointment = scheduler.session.query(Appointment).filter_by(
                            patient_id=patient_id,
                            doctor_id=doctor_id
                        ).first()
                        if not appointment:
                            return {
                                "success": False,
                                "message": f"No appointment found for patient ID {patient_id}"
                            }
                        
                        # Update the appointment datetime
                        appointment.datetime = new_datetime_obj
                    
                        # Commit the changes
                        scheduler.session.commit()
                print("Got here")
                print(appointment.datetime)
                return {
                    "type": "appointment_result",
                    "success": True,
                    "message": "Appointment managed successfully",
                    "appointment": {
                        "id": appointment.appointment_id,
                        "doctor_name": doctor.name if doctor else "Unknown",
                        "patient_name": patient.name if patient else "Unknown",
                        "datetime": appointment.datetime.strftime('%Y-%m-%d %H:%M'),
                        "status": appointment.status
                    }
                }
            except Exception as e:
                logger.error(f"Error ending appointment conversation: {str(e)}")
                return {
                    "type": "appointment_result",
                    "success": False,
                    "message": f"Failed to schedule appointment: {str(e)}"
                }
        elif agent.message_type == 'medication':
            try:
                # Generate a summary of the medication conversation
                context = agent.format_conversation(agent.LLM.conversation_history)
                summary_prompt = "Summarize the medication adherence from this conversation. Did the patient take their medication? Extract any relevant details about side effects or concerns."
                summary = agent.moremi_response(context, summary_prompt)
                
                return {
                    "type": "medication_result",
                    "success": True,
                    "summary": summary
                }
            except Exception as e:
                logger.error(f"Error ending medication conversation: {str(e)}")
                return {
                    "type": "medication_result",
                    "success": False,
                    "message": str(e)
                }
        else:
            return {
                "type": "unknown_result",
                "success": False,
                "message": f"Unknown conversation type: {agent.message_type}"
            }

    @staticmethod
    def process_appointment_reminders(scheduler, hours_ahead, days_ahead):
        """Process appointment reminders using the scheduler"""
        try:
            logger.info("Processing appointment reminders")
            # Call the scheduler's method to get appointment reminders
            reminders = scheduler.get_upcoming_appointments(hours_ahead=hours_ahead)
            # Create reminder messages
            reminder_messages = []
            for appointment, patient, doctor in reminders:
                logger.info(f'Processing appointment for {patient.name}')
                reminder = ReminderMessage(
                    patient_name=patient.name,
                    message_type='appointment',
                    details={
                        'doctor_id': doctor.doctor_id,
                        'doctor_name': doctor.name,
                        'datetime': appointment.datetime,
                        'status': appointment.status,
                        'patient_id': patient.patient_id
                    }
                )
                reminder_messages.append(reminder)
            
            return reminder_messages
        except Exception as e:
            logger.error(f"Error processing appointment reminders: {str(e)}")
            return []
    
    @staticmethod
    def process_medication_reminders(scheduler, days_ahead):
        """Process medication reminders using the scheduler"""
        try:
            logger.info("Processing medication reminders")
            # Get active medications
            medications = scheduler.get_active_medications()
            logger.info(f"Retrieved {len(medications)} active medications")
            
            # Create reminder messages
            reminder_messages = []
            for medication, patient in medications:
                logger.info(f"Processing medication {medication.name} for patient {patient.name}")
                reminder = ReminderMessage(
                    patient_name=patient.name,
                    message_type='medication',
                    details={
                        'medication_name': medication.name,
                        'dosage': medication.dosage,
                        'frequency': medication.frequency
                    }
                )
                reminder_messages.append(reminder)
            
            return reminder_messages
        except Exception as e:
            logger.error(f"Error processing medication reminders: {str(e)}")
            return []

class SchedulerRequest(BaseModel):
    hours_ahead: int
    days_ahead: int 
    reminder_type: str = Field(..., alias="type")  # Accept 'type' from frontend but use 'reminder_type' internally

class ConversationRequest(BaseModel):
    patient_id: int
    reminder_type: str
    details: Dict[str, Any] = {}

class ConversationResponse(BaseModel):
    status: str
    message: str
    conversation_id: Optional[str] = None

class ReminderResponse(BaseModel):
    id: str
    patient_name: str
    message_type: str
    details: Dict[str, Any]
    timestamp: datetime

class AppointmentRequest(BaseModel):
    doctor_id: int
    patient_id: int
    datetime: Optional[str] = None
    days_ahead: int

goodbye_messages = [
  "Have a great day!",
  "Have a wonderful day!",
  "Take care!",
  "I hope this helps!",
  "Best wishes for your health!",
  "I'm glad I could assist you.",
  "Stay healthy and happy!",
  "I'm always here if you need me.",
  "It was a pleasure helping you.",
  "Goodbye for now!",
  "I hope you feel better soon!",
  "Goodbye",
  "Bye",
  "Enjoy your day!"
]

@app.post("/start-scheduler")
async def start_scheduler(request: SchedulerRequest):
    """
    Start the scheduler for reminders
    """
    try:
        logger.info(f"Starting scheduler with: hours_ahead={request.hours_ahead}, days_ahead={request.days_ahead}, reminder_type={request.reminder_type}")
        
        # Create a scheduler manager
        with SchedulerManager(days_ahead=request.days_ahead) as scheduler:
            # Process reminders based on type
            if request.reminder_type == 'appointment':
                reminders = ReminderService.process_appointment_reminders(
                    scheduler, hours_ahead=request.hours_ahead, days_ahead=request.days_ahead
                )
            else:  # medication
                # Note: process_medication_reminders only accepts days_ahead
                reminders = ReminderService.process_medication_reminders(
                    scheduler, days_ahead=request.days_ahead
                )
            
            # Convert reminders to response format and store them in our service
            response_reminders = []
            for reminder in reminders:
                # Create a unique ID for this reminder
                reminder_id = str(uuid.uuid4())
                
                # Create reminder message and store it in our service
                reminder_message = ReminderMessage(
                    patient_name=reminder.patient_name,
                    message_type=request.reminder_type,
                    details=reminder.details
                )
                
                # Store the reminder in our centralized service
                active_reminders[reminder_id] = reminder_message
                
                # Create the agents
                agent = ReminderService.create_or_get_agent(reminder_id, request.days_ahead)
                
                # Add to response
                response_reminders.append(
                    ReminderResponse(
                        id=reminder_id,
                        patient_name=reminder.patient_name,
                        message_type=request.reminder_type,
                        details=reminder.details,
                        timestamp=datetime.now()
                    )
                )
            
            logger.info(f"Processed {len(reminders)} {request.reminder_type} reminders")
            
            return response_reminders
    except Exception as e:
        logger.error(f"Error starting scheduler: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{reminder_id}")
async def websocket_endpoint(websocket: WebSocket, reminder_id: str):
    await websocket.accept()
    
    logger.info(f"WebSocket connection accepted for reminder ID: {reminder_id}")
    
    # Create a ping task to keep the connection alive
    ping_task = None
    
    async def send_ping():
        while True:
            try:
                await asyncio.sleep(30)  # Send a ping every 30 seconds
                await websocket.send_json({"type": "ping"})
                logger.debug(f"Sent ping to client {reminder_id}")
            except Exception as e:
                logger.error(f"Error in ping task: {str(e)}")
                break
    
    # Start the ping task
    ping_task = asyncio.create_task(send_ping())
    
    try:
        # Initialize voice assistant and medical agent
        voice_assistant = ReminderService.create_or_get_voice_assistant(reminder_id)
        agent = ReminderService.create_or_get_agent(reminder_id)
        
        # Wait for initial context
        context = await websocket.receive_json()
        logger.info(f"Received initial context for reminder ID: {reminder_id}")
        
        if context['type'] == 'context':
            try:
                reminder_data = json.loads(context['context'])
                logger.info(f"Parsed reminder data: {reminder_data}")
                
                # Check if we have a reminder with this ID
                reminder = ReminderService.get_reminder(reminder_id)
                if not reminder:
                    # Create a new reminder if one doesn't exist
                    logger.info(f"Creating new reminder for ID: {reminder_id}")
                    # Store the reminder with the provided reminder_id
                    active_reminders[reminder_id] = ReminderMessage(
                        patient_name=reminder_data.get('patient_name', 'Patient'),
                        message_type=reminder_data.get('message_type', 'medication'),
                        details=reminder_data.get('details', {})
                    )
                    # Re-get the agent with the new reminder
                    agent = ReminderService.create_or_get_agent(
                        reminder_id, 
                        days_ahead=reminder_data.get('details', {}).get('days_ahead', 30)
                    )
                
                # Generate initial message
                initial_message = ReminderService.generate_initial_message(reminder_id)
                
                # Send initial message
                await websocket.send_json({
                    "type": "message",
                    "text": initial_message
                })
                
                # Use TTS client directly for the initial message
                tts_client.TTS(
                    initial_message,
                    play_locally=True
                )
                
                tts_client.wait_for_completion()
            except Exception as e:
                logger.error(f"Error processing context: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing context: {str(e)}"
                })
        
        # Process incoming messages
        while True:
            try:
                data = await websocket.receive_json()
                
                if data.get("type") == "message":
                    try:
                        # Process the message with the agent
                        logger.info(f"Processing message: {data['text']}")
                        response = agent.moremi_response(data['text'], agent.system_prompt)
                        logger.info(f"Response: {response}")
                        
                        # Check if this is a reschedule request
                        if 'reschedule_appointment' in response:
                            logger.info("User requested to reschedule appointment")
                            # Get available slots
                            reschedule_msg = agent.reschedule_message
                            available_slots = agent.appointment_rescheduler()
                            
                            #Format the available slots
                            slots_response = agent.moremi_response(f"Reword the information on available doctor slots here {available_slots} into a user friendly message, Output the dates as example: 29th March 2025 at 04:45 PM", "You are a helpful assistant capable of summarizing and rewording text")
                            agent.LLM.conversation_history[-2]['role'] = "assistant"
                            agent.LLM.conversation_history[-2]['content'] = f"These are the available slots I found from the database, {available_slots}, I should format into a user friendly message and send to the user"
                            
                            # Send reschedule message
                            await websocket.send_json({
                                'type': 'message',
                                'text': reschedule_msg + "\n" + slots_response
                            })
                            
                            # Use TTS for the reschedule message
                            tts_client.TTS(
                                reschedule_msg + "\n" + slots_response,
                                play_locally=True
                            )
                            tts_client.wait_for_completion()
                            
                            # Don't break - allow user to continue conversation
                            continue
                        
                        # Check if this is a goodbye message
                        elif any(goodbye_message.strip('.!?').lower() in response.lower() for goodbye_message in goodbye_messages):
                            logger.info("Conversation is ending")
                            try:    
                                # Send goodbye message
                                await websocket.send_json({
                                    'type': 'message',
                                    'text': response,
                                    'is_goodbye': True
                                })
                                
                                try:
                                    # Use TTS for the confirmation
                                    tts_client.TTS(
                                        response,
                                        play_locally=True
                                    )
                                    tts_client.wait_for_completion()
                                except KeyboardInterrupt:
                                    tts_client.stop()
                                    pass
                                
                                # Use our centralized service to end the conversation
                                result = ReminderService.end_conversation(reminder_id)
                                
                                # Send the result to the client
                                if result.get('type') == 'appointment_result':
                                    # Prepare a user-friendly message
                                    if result.get('success', False):
                                        appointment = result.get('appointment', {})
                                        appointment_datetime = appointment.get('datetime', '')
                                        if appointment_datetime:
                                            try:
                                                # Convert ISO format to readable format
                                                dt = datetime.fromisoformat(appointment_datetime)
                                                formatted_datetime = dt.strftime('%B %d at %I:%M %p')
                                            except:
                                                formatted_datetime = appointment_datetime
                                        else:
                                            formatted_datetime = "the scheduled time"
                                        
                                        confirmation = f"Great! Your appointment has been scheduled for {formatted_datetime}."
                                        
                                        # Send confirmation message
                                        await websocket.send_json({
                                            'type': 'message',
                                            'text': confirmation
                                        })
                                        
                                        # Use TTS for the confirmation
                                        tts_client.TTS(
                                            confirmation,
                                            play_locally=True
                                        )
                                        tts_client.wait_for_completion()
                                    else:
                                        error_msg = f"I couldn't schedule the appointment: {result.get('message', 'Unknown error')}"
                                        
                                        # Send error message
                                        await websocket.send_json({
                                            'type': 'message',
                                            'text': error_msg
                                        })
                                        
                                        # Use TTS for the error message
                                        tts_client.TTS(
                                            error_msg,
                                            play_locally=True
                                        )
                                        tts_client.wait_for_completion()
                                    
                                    await asyncio.sleep(1)
                                    # Send the detailed result
                                    await websocket.send_json({
                                        'type': 'appointment_result',
                                        'success': result.get('success', False),
                                        'message': result.get('message', ''),
                                        'appointment': result.get('appointment', {})
                                    })
                                elif result.get('type') == 'medication_result':
                                    # Prepare a user-friendly message
                                    if result.get('success', False):
                                        summary_msg = "Thank you for your time. Your medication information has been updated."
                                    else:
                                        summary_msg = "Thank you for your time. I couldn't generate a summary of our conversation."
                                    
                                    # Send summary message
                                    await websocket.send_json({
                                        'type': 'message',
                                        'text': summary_msg
                                    })
                                    
                                    # Use TTS for the summary message
                                    tts_client.TTS(
                                        summary_msg,
                                        play_locally=True
                                    )
                                    tts_client.wait_for_completion()
                                    
                                    # Send the detailed result
                                    await websocket.send_json({
                                        'type': 'medication_result',
                                        'success': result.get('success', False),
                                        'summary': result.get('summary', '')
                                    })
                                
                                # Break the loop to close the connection
                                #break
                            except Exception as e:
                                logger.error(f"Error processing end_conversation: {str(e)}")
                                await websocket.send_json({
                                    'type': 'message',
                                    'text': f"An error occurred while ending the conversation: {str(e)}"
                                })
                                break    
                    
                        elif data.get("type") == "end_conversation":
                            try:
                                logger.info(f"Received end_conversation request for {reminder_id}")
                                
                                # Use our centralized service to end the conversation
                                result = ReminderService.end_conversation(reminder_id)
                                
                                # Send the result to the client
                                if result.get('type') == 'appointment_result':
                                    # Prepare a user-friendly message
                                    if result.get('success', False):
                                        appointment = result.get('appointment', {})
                                        appointment_datetime = appointment.get('datetime', '')
                                        if appointment_datetime:
                                            try:
                                                # Convert ISO format to readable format
                                                dt = datetime.fromisoformat(appointment_datetime)
                                                formatted_datetime = dt.strftime('%B %d at %I:%M %p')
                                            except:
                                                formatted_datetime = appointment_datetime
                                        else:
                                            formatted_datetime = "the scheduled time"
                                        
                                        confirmation = f"Great! Your appointment has been scheduled for {formatted_datetime}."
                                        
                                        # Send confirmation message
                                        await websocket.send_json({
                                            'type': 'message',
                                            'text': confirmation
                                        })
                                        
                                        # Use TTS for the confirmation
                                        tts_client.TTS(
                                            confirmation,
                                            play_locally=True
                                        )
                                        tts_client.wait_for_completion()
                                    else:
                                        error_msg = f"I couldn't schedule the appointment: {result.get('message', 'Unknown error')}"
                                        
                                        # Send error message
                                        await websocket.send_json({
                                            'type': 'message',
                                            'text': error_msg
                                        })
                                        
                                        # Use TTS for the error message
                                        tts_client.TTS(
                                            error_msg,
                                            play_locally=True
                                        )
                                        tts_client.wait_for_completion()
                                    
                                    # Send the detailed result
                                    await websocket.send_json({
                                        'type': 'appointment_result',
                                        'success': result.get('success', False),
                                        'message': result.get('message', ''),
                                        'appointment': result.get('appointment', {})
                                    })
                                elif result.get('type') == 'medication_result':
                                    # Prepare a user-friendly message
                                    if result.get('success', False):
                                        summary_msg = "Thank you for your time. Your medication information has been updated."
                                    else:
                                        summary_msg = "Thank you for your time. I couldn't generate a summary of our conversation."
                                    
                                    # Send summary message
                                    await websocket.send_json({
                                        'type': 'message',
                                        'text': summary_msg
                                    })
                                    
                                    # Use TTS for the summary message
                                    tts_client.TTS(
                                        summary_msg,
                                        play_locally=True
                                    )
                                    tts_client.wait_for_completion()
                                    
                                    # Send the detailed result
                                    await websocket.send_json({
                                        'type': 'medication_result',
                                        'success': result.get('success', False),
                                        'summary': result.get('summary', '')
                                    })
                                
                                # Break the loop to close the connection
                                #break
                        
                            except Exception as e:
                                logger.error(f"Error processing end_conversation: {str(e)}")
                                await websocket.send_json({
                                    'type': 'message',
                                    'text': f"An error occurred while ending the conversation: {str(e)}"
                                })
                                break
                            
                            
                        # Handle ping messages from client
                        elif data.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                            logger.debug(f"Received ping from client {reminder_id}, sent pong")
                            continue
                        
                        # Handle pong responses from client
                        elif data.get("type") == "pong":
                            logger.debug(f"Received pong from client {reminder_id}")
                            continue
                        
                        
                        #Regular Response
                        else:
                            logger.info("Regular response")
                            
                            # Send message
                            await websocket.send_json({
                                'type': 'message',
                                'text': response
                            })
                            
                            # Use TTS for the confirmation
                            tts_client.TTS(
                                response,
                                play_locally=True
                            )
                            tts_client.wait_for_completion()
                            
                            # Don't break - allow user to continue conversation
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error in conversation: {str(e)}")
                        await websocket.send_json({
                            'type': 'message',
                            'text': f"An error occurred: {str(e)}"
                        })
                    
            except WebSocketDisconnect:
                logger.info(f"Client disconnected: {reminder_id}")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                break
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for reminder {reminder_id}")
    except Exception as e:
        logger.error(f"Error in medication websocket: {str(e)}")
        error_message = str(e)
        await websocket.send_json({
            "type": "message",
            "text": f"An error occurred: {error_message}"
        })
        
        # Use TTS client directly for error messages
        tts_client.TTS(
            f"An error occurred: {error_message}",
            play_locally=True
        )
        
        tts_client.wait_for_completion()
        
    finally:
        # Cancel the ping task when the connection is closed
        if ping_task:
            ping_task.cancel()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
