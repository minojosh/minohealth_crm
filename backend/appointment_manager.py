from .database import Patient, Appointment, Medication, Doctor, Session, DoctorWorkingDay
from sqlalchemy import and_
from datetime import datetime, timedelta
import logging
import os
import sys
from .STT_client import SpeechRecognitionClient
from .XTTS_adapter import TTSClient
from .config import DATABASE_URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, func
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from .moremi import ConversationManager
import base64
import sounddevice as sd
import numpy as np
import requests
import json
import time
import re
import argparse
from .config import moremi
from dotenv import load_dotenv
import asyncio

# Initialize clients
client = SpeechRecognitionClient()
load_dotenv()
# Initialize TTSClient with speech service URL
clientjosh = TTSClient(os.getenv("SPEECH_SERVICE_URL"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('appointment_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Message:
    user_input: str
    response: Optional[str] = None

    def add_response(self, response: str):
        self.response = response

    def to_dict(self):
        return asdict(self)

@dataclass
class ReminderMessage:
    """Represents a reminder message for appointments or medications."""
    patient_name: str
    message_type: str  # 'appointment' or 'medication'
    details: dict
    timestamp: datetime = datetime.now()
    
@dataclass
class ReminderResult:
    """Result of processing a reminder."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    
goodbye_messages = [
  "Have a great day!",
  "Take care!",
  "I hope this helps!",
  "Best wishes for your health!",
  "I'm glad I could assist you.",
  "Stay healthy and happy!",
  "I'm always here if you need me.",
  "It was a pleasure helping you.",
  "Goodbye for now!",
  "I hope you feel better soon!"
]

class VoiceAssistant:
    """Handles speech recognition and synthesis for reminders using TTSClient."""
    
    SILENCE_TIMEOUT = 4  # seconds - increased from 3 to 4
    
    def __init__(self):
        """Initialize the voice assistant with TTSClient and STTClient."""
        self.tts_client = TTSClient(api_url=os.getenv("XTTS_URL") or os.getenv("SPEECH_SERVICE_URL"))
        self.stt_client = SpeechRecognitionClient()
        self.last_audio_data = None
        self.last_audio_base64 = None
        self.sample_rate = 16000  # Default sample rate

    def get_last_audio_base64(self) -> Optional[str]:
        """Get the last synthesized audio as base64 for frontend playback."""
        return self.last_audio_base64
            
    def recognize_speech(self) -> str:
        """Record and transcribe speech using STT client."""
        logger.info("Starting speech recognition with STT client...")
        try:
            # Use the STT client to recognize speech
            transcription = self.stt_client.recognize_speech()
            logger.info(f"Transcription result: {transcription}")
            return transcription
        except Exception as e:
            logger.error(f"Error in speech recognition: {str(e)}")
            return ""

    def synthesize_speech(self, text: str) -> None:
        """Synthesize text to speech using TTSClient."""
        try:
            logger.info("Synthesizing speech with TTSClient...")
            
            # Use the TTSClient to generate speech
            audio_data, sample_rate, base64_audio, _ = self.tts_client.TTS(
                text, 
                play_locally=False,
                return_data=True
            )
            
            # Store the data for potential frontend playback
            self.last_audio_data = audio_data
            self.last_audio_base64 = base64_audio
            self.sample_rate = sample_rate
            
            logger.info("Speech synthesis complete")
        except Exception as e:
            logger.error(f"Error in speech synthesis: {str(e)}")
            self.last_audio_data = None
            self.last_audio_base64 = None

class SchedulerManager:
    """Manages appointment and medication scheduling."""

    def __init__(self, days_ahead):
        # Create engine and session
        self.engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.days_ahead = days_ahead
        logger.info(f"Initialized SchedulerManager with database at {DATABASE_URL}")
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()

    def get_upcoming_appointments(self, hours_ahead: int = 24) -> List[Tuple[Appointment, Patient, Doctor]]:
        """Retrieve upcoming appointments within specified hours."""
        now = datetime.now()
        next_time = now + timedelta(hours=hours_ahead)
        logger.info(f"Looking for appointments between {now} and {next_time}")
        
        try:
            appointments = self.session.query(Appointment, Patient, Doctor).join(Patient).join(Doctor).filter(
                and_(
                    Appointment.datetime >= now,
                    Appointment.datetime <= next_time,
                    Appointment.status == 'scheduled'
                )
            ).all()
            
            logger.info(f"Found {len(appointments)} upcoming appointments")
            for appt, patient, doctor in appointments:
                logger.debug(f"Appointment: {patient.name} with Dr. {doctor.name} at {appt.datetime}")
            
            return appointments
            
        except Exception as e:
            logger.error(f"Error retrieving upcoming appointments: {e}")
            return []

    def get_active_medications(self) -> List[Tuple[Medication, Patient]]:
        """Retrieve active medications for all patients."""
        now = datetime.now()
        try:
            logger.info(f"Querying active medications for date: {now.date()}")
            medications = self.session.query(Medication, Patient).join(Patient).filter(
                and_(
                    Medication.start_date <= now.date(),
                    Medication.end_date >= now.date()
                )
            ).all()
            
            logger.info(f"Found {len(medications)} active medications")
            # Debug logging
            for med, patient in medications:
                logger.info(f"Found medication: {med.name} for patient: {patient.name}")
                logger.debug(f"Medication details - Start: {med.start_date}, End: {med.end_date}, Dosage: {med.dosage}")
            
            return medications
            
        except Exception as e:
            logger.error(f"Error retrieving active medications: {e}")
            return []

    # ...existing code...

    def _get_new_date(self, doctor_id: int) -> List[dict]:
        """
        Find available appointment slots for a doctor within the specified number of days ahead.
        """
        available_slots = []
        now = datetime.now()
        end_date = now + timedelta(days=self.days_ahead)
        
        # Get doctor's working days and hours
        working_days = self.session.query(DoctorWorkingDay).filter(
            DoctorWorkingDay.doctor_id == doctor_id
        ).all()
        
        # Get existing appointments
        existing_appointments = self.session.query(Appointment).filter(
            and_(
                Appointment.doctor_id == doctor_id,
                Appointment.datetime >= now,
                Appointment.datetime <= end_date,
                Appointment.status == 'scheduled'
            )
        ).all()
        
        # Convert existing appointments to a set of datetime strings for quick lookup
        booked_slots = {appt.datetime.strftime('%Y-%m-%d %H:%M') for appt in existing_appointments}
        
        # For each working day, generate available slots
        current_date = now.date()
        while current_date <= end_date.date():
            day_of_week = current_date.strftime('%A').lower()  # Get day name in lowercase
            
            # Find working hours for this day
            day_schedule = next((wd for wd in working_days if wd.day_of_week.lower() == day_of_week), None)
            
            if day_schedule:
                # start_time and end_time are already datetime.time objects
                start_time = day_schedule.start_time
                end_time = day_schedule.end_time
                
                # Generate slots at 30-minute intervals
                current_time = datetime.combine(current_date, start_time)
                slot_end = datetime.combine(current_date, end_time)
                
                while current_time < slot_end:
                    # Skip slots in the past
                    if current_time > now:
                        slot_str = current_time.strftime('%Y-%m-%d %H:%M')
                        # Check if slot is not already booked
                        if slot_str not in booked_slots:
                            available_slots.append({
                                'date': current_time.strftime('%Y-%m-%d'),
                                'time': current_time.strftime('%I:%M %p'),
                                'day_of_week': current_time.strftime('%A'),
                                'datetime': current_time
                            })
                    
                    current_time += timedelta(minutes=30)
            
            current_date += timedelta(days=1)  # Move to next day

        return available_slots

    def format_available_slots(self, slots: List[dict], doctor_name: Optional[str] = None) -> str:
        """Format available time slots in a human-friendly way, grouped by date."""
        if not slots:
            return f"I'm sorry, but there are no available appointment slots in the next {self.days_ahead} days."
        
        # Group slots by date
        slots_by_date = {}
        for slot in slots:
            date_key = slot['date']
            if date_key not in slots_by_date:
                slots_by_date[date_key] = []
            slots_by_date[date_key].append(slot['time'])
        
        # Format the message
        if doctor_name is None:
            message_parts = [f"Here are the available appointment slots in the next {self.days_ahead} days:"]
        else:
            message_parts = [f"Here are the available appointment slots for {doctor_name} in the next {self.days_ahead} days:"]
            
        for date_key in sorted(slots_by_date.keys()):
            # Convert date string to datetime for better formatting
            date_obj = datetime.strptime(date_key, '%Y-%m-%d')
            date_header = date_obj.strftime('%A, %B %d, %Y')
            
            times = slots_by_date[date_key]
            start_time = times[0]
            end_time = times[-1] 
            
            message_parts.append(f"\n{date_header} from {start_time} to {end_time}")

        
        message_parts.append("\nWhich day and time would you prefer for a 30 minutes appointment?")
        
        return '\n'.join(message_parts)

    def process_appointment_reminders(self, hours_ahead, days_ahead) -> List[ReminderMessage]:
        """Process and create reminders for upcoming appointments."""
        reminders = []
        try:
            logger.info("retrieving upcoming appointment")
            appointments = self.get_upcoming_appointments(hours_ahead=hours_ahead)
            for appointment, patient, doctor in appointments:
                logger.info(f'Processing appointment for {patient.name}')
                reminder = ReminderMessage(
                    patient_name=patient.name,
                    message_type='appointment',
                    details={
                        'doctor_id': doctor.doctor_id,  # Add doctor_id to details
                        'doctor_name': doctor.name,
                        'datetime': appointment.datetime,
                        'status': appointment.status,
                        'patient_id': patient.patient_id
                    }
                )
                reminders.append(reminder)
        except Exception as e:
            logger.error(f"Error processing appointment reminders: {str(e)}")
        return reminders

    def process_one_medication_reminder(self, medication: Medication, patient: Patient) -> ReminderResult:
        """Process a single medication reminder and return the result."""
        try:
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
            conversation = MedicalAgent(reminder, reminder.message_type, self.days_ahead)
            logger.info('Starting conversation for medication reminder')
            result = conversation.conversation_manager()
            
            return ReminderResult(
                success=True,
                message=f"Successfully processed medication reminder for {patient.name}",
                data={
                    "patient_name": patient.name,
                    "medication_name": medication.name,
                    "adherence": "Confirmed" if result else "Unknown"
                }
            )
        except Exception as e:
            logger.error(f"Error processing medication reminder: {e}")
            return ReminderResult(
                success=False,
                message=f"Failed to process medication reminder: {str(e)}"
            )
   
    def process_medication_reminders(self, days_ahead) -> List[ReminderMessage]:
        """Process and create reminders for active medications."""
        reminders = []
        results = []
        try:
            logger.info("Retrieving active medications...")
            medications = self.get_active_medications()
            logger.info(f"Retrieved {len(medications)} active medications")
            
            # First, create all reminder objects
            for medication, patient in medications:
                reminder = ReminderMessage(
                    patient_name=patient.name,
                    message_type='medication',
                    details={
                        'medication_name': medication.name,
                        'dosage': medication.dosage,
                        'frequency': medication.frequency
                    }
                )
                reminders.append(reminder)
            
            # Now, process one medication at a time - sequentially
            # (Only process the first one if specified for testing)
            if len(medications) > 0:
                medication, patient = medications[0]  # Just process the first one for testing
                result = self.process_one_medication_reminder(medication, patient)
                results.append(result)
                logger.info(f"Processed 1 of {len(medications)} medication reminders")
        except Exception as e:
            logger.error(f"Error in process_medication_reminders: {e}")
        
        return reminders

class MedicalAgent:
    def __init__(self, reminder, message_type, days_ahead):
        self.scheduler = SchedulerManager(days_ahead)
        self.voice_assistant = VoiceAssistant()
        self.reminder = reminder
        self.message_type = message_type
        self.api = "http://46.101.91.139:5001/inference_history" #os.getenv('API_URL')
        self._load_prompts()
        self.audio_base64 = None
        
        # Set default values if reminder is None
        if self.reminder is None:
            self.reminder = type('DummyReminder', (), {
                'patient_name': 'Patient',
                'details': {
                    'patient_id': 1,
                    'doctor_id': 1,
                    'doctor_name': 'Doctor',
                    'datetime': datetime.now() + timedelta(days=7),
                    'medication_name': 'your medication',
                    'dosage': 'as prescribed',
                    'frequency': 'as directed'
                }
            })
            logger.warning(f"Created dummy reminder for {message_type} conversation")

    def _load_prompts(self):
        try:
            with open('backend/call_prompts.json', 'r', encoding='utf-8') as f:
                prompts=json.load(f)
                self.initial_appointment_message=prompts['initial_appointment_reminder']
                self.initial_medication_message=prompts['initial_medication_reminder']
                self.medication_prompt=prompts['medication_prompt']
                self.appointment_prompt=prompts['appointment_prompt']
                self.extract_prompt=prompts['extract_prompt']
                self.reschedule_message=prompts.get('reschedule_message', "Let me find available slots for rescheduling.")
        except FileNotFoundError:
            logger.warning("call_prompts.json not found. Using default empty prompts.")
            self.initial_appointment_message=""
            self.initial_medication_message=""
            self.system_prompt=""
            self.reschedule_message="Let me find available slots for rescheduling."
    
    def moremi_response(self, messages, system_prompt: Optional[str] = None) -> str:
        """Get response from Moremi API using ConversationManager for text-only processing."""
        
        
        base_url = "http://46.101.91.139:5003/v1"  
        conversation_manager = ConversationManager(base_url=base_url)
        print("yaaayyyyy")
        
        
        conversation_manager.custom_params["system_prompt"] = system_prompt
        
        
        conversation_manager.add_user_message(text=messages)

        
    
        response = conversation_manager.get_assistant_response()
        
        return response

    def appointment_rescheduler(self):
        """Reschedules an appointment."""
        # Search for available slots
        try:
            logger.info(f"Rescheduling appointment for {self.reminder.patient_name}")
            available_slots = self.scheduler._get_new_date(self.reminder.details['doctor_id'])  # Uses scheduler.days_ahead
            if available_slots:
                return self.scheduler.format_available_slots(available_slots)
            else:
                return "I am sorry, but there are no available appointment slots in the next 30 days."
        except Exception as e:
            logger.error(f"Error rescheduling appointment: {e}")
            raise
    
    def format_conversation(self, messages):
        """
        Convert a list of Message objects into a formatted text dialogue
        between Patient and Doctor.
        """
        formatted_text = "Medical Appointment Conversation\n"
        formatted_text += "=" * 30 + "\n\n"
        
        for msg in messages:
            # Format patient's message
            if msg.user_input:
                formatted_text += f"Patient: {msg.user_input}\n\n"
            
            # Format doctor's message
            if msg.response:
                # Clean up any markdown or special formatting
                clean_response = msg.response.strip()
                formatted_text += f"Doctor: {clean_response}\n\n"
        
        return formatted_text

    def extract_datetime_from_moremi(self, response: str) -> str:
        pattern = r"""
            # Match weekday
            ([A-Za-z]+day)[\s,]*
            # Match month and day
            ([A-Za-z]+\s+\d{1,2})
            # Optional year
            (?:,?\s+(\d{4}))?
            # Optional 'at' or comma before time
            (?:,?\s+(?:at\s+)?|\s+at\s+)?
            # Match time
            (\d{1,2}:\d{2}\s*[APM]{2})
        """
        
        match = re.search(pattern, response, re.VERBOSE)
        if not match:
            raise ValueError("Could not find datetime information in text")
        
        weekday = match.group(1)
        date = match.group(2)
        year = match.group(3) if match.group(3) else "2025"
        time = match.group(4)

        datetime_str = f"{weekday}, {date}, {year} at {time}"
        
        try:
            # Convert to datetime object
            dt = datetime.strptime(datetime_str, "%A, %B %d, %Y at %I:%M %p")
            # Format the datetime object to desired output format
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError as e:
            raise ValueError(f"Failed to parse datetime: {e}")

    def conversation_manager(self):
        messages = []
        if self.message_type == 'appointment':
            self.system_prompt = self.appointment_prompt
            initial_message = self.initial_appointment_message.format(
                patient_name=self.reminder.patient_name,
                doctor_name=self.reminder.details['doctor_name'],
                appointment_datetime=self.reminder.details['datetime'].strftime('%B %d at %I:%M %p')
            )
        elif self.message_type == 'medication':
            self.system_prompt = self.medication_prompt.format(
                medication_dosage=self.reminder.details['dosage'],
                medication_frequency=self.reminder.details['frequency']
            )
            initial_message = self.initial_medication_message.format(
                patient_name=self.reminder.patient_name,
                medication_name=self.reminder.details['medication_name'],
                medication_dosage=self.reminder.details['dosage'],
                medication_frequency=self.reminder.details['frequency'],
            )

        messages.append(Message(user_input='Hi', response=initial_message))
        print("speaking........")
        
        # Use TTSClient exclusively for speech synthesis
        audio_data, sample_rate, self.audio_base64 = clientjosh.TTS(initial_message, play_locally=True, return_data=True)
        print("speaking done.......")
        
        try:
            while True:
                # Get speech input using STT client
                # Get speech input
                client = SpeechRecognitionClient()  # Reinitialize client each time
                client.recognize_speech()
                user_input = client.get_transcription()
                
                if not user_input or not user_input.strip():
                    logger.warning("Empty input received. Please say something.")
                    continue
                    
                logger.info(f"User input: {user_input}")
                
                # Get Moremi response
                messages.append(Message(user_input=user_input))
                response = self.moremi_response(messages, self.system_prompt)
                
                if 'reschedule_appointment' in response:
                    logger.info("User requested to reschedule appointment")
                    # Use TTSClient for speech synthesis
                    audio_data, sample_rate, base64_audio = clientjosh.TTS(
                        self.reschedule_message, 
                        play_locally=True,
                        return_data=True
                    )
                    self.audio_base64 = base64_audio
                    response = self.appointment_rescheduler()
                elif any(goodbye_message.strip('.!?').lower() in response.lower() for goodbye_message in goodbye_messages):
                    messages[-1].add_response(response)
                    logger.info(f"Moremi response: {response}")
                    logger.info("Conversation is over")
                    
                    # Use TTSClient for speech synthesis
                    audio_data, sample_rate, base64_audio = clientjosh.TTS(
                        response, 
                        play_locally=True,
                        return_data=True
                    )
                    self.audio_base64 = base64_audio
                    return True
                else:
                    # User confirms request
                    pass
                    
                messages[-1].add_response(response)
                logger.info(f"Moremi response: {response}")
               
                # Use TTSClient for speech synthesis
                audio_data, sample_rate, base64_audio = clientjosh.TTS(
                    response, 
                    play_locally=True,
                    return_data=True
                )
                self.audio_base64 = base64_audio
                print("\nReady for next input...")

        except KeyboardInterrupt:
            logger.info("Conversation ended by user")
            print('\nEnding conversation...') 
            return False     
        except Exception as e:
            logger.error(f"Error in conversation loop: {str(e)}")
            error_response = 'Sorry. The system is currently unavailable.'
            try:
                # Use TTSClient for error message
                clientjosh.TTS(error_response, play_locally=True)
            except:
                pass
            return False
    
        context = self.format_conversation(messages)
        final = moremi(context, self.extract_prompt) 
        print(f'final:{final}')
        day = self.extract_datetime_from_moremi(final)
        print(f'datetime: {day}')
        newday = datetime.strptime(day, '%Y-%m-%d %H:%M')
        print(schedule_appointment(self.reminder.details['patient_id'], newday))
        return True

def main():
    """Main function to run the scheduler."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the appointment and medication scheduler.')
    parser.add_argument('--hours-ahead', type=int, required=True,
                      help='Number of hours ahead to check for appointments')
    parser.add_argument('--days-ahead', type=int, required=True,
                      help='Number of days ahead to look for available slots when rescheduling')
    parser.add_argument('--type', type=str, required=True,
                      help='Type of reminder to process (appointment or medication)')
                                      
    args = parser.parse_args()
    
    logger.info("Starting appointment and medication scheduler")
    
    try:
        with SchedulerManager(args.days_ahead) as scheduler:
            if args.type == 'appointment':
                # Process appointment reminders with both hours_ahead and days_ahead
                appointment_reminders = scheduler.process_appointment_reminders(
                    hours_ahead=args.hours_ahead,
                    days_ahead=args.days_ahead
                )
                logger.info(f"Processed {len(appointment_reminders)} appointment reminders")
            else:
                # Process medication reminders - only process one at a time
                medication_reminders = scheduler.process_medication_reminders(days_ahead=args.days_ahead)
                logger.info(f"Processed medication reminders")
                
    except Exception as e:
        logger.error(f"Error in scheduler main function: {e}")
        raise

if __name__ == "__main__":
    main()
