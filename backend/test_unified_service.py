#!/usr/bin/env python3
"""
Test script for the unified AI service implementation.
This script demonstrates how to use the different services for various medical CRM tasks.
"""

import os
import sys
from pathlib import Path
import logging

# Add the parent directory to the path to import the unified_ai_service module
sys.path.append(str(Path(__file__).parent.parent))

# Import the unified service components
from backend.unified_ai_service import (
    ConfigManager,
    SchedulerService,
    DifferentialDiagnosisService,
    MedicationReminderService,
    SpeechAssistant
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_config_manager():
    """Test the ConfigManager class."""
    print("\n=== Testing ConfigManager ===")
    config = ConfigManager()
    
    # Test loading prompts
    system_prompt = config.get_prompt("systemprompt")
    print(f"System prompt loaded: {bool(system_prompt)}")
    print(f"System prompt preview: {system_prompt[:100]}...")
    
    # Test loading environment variables
    print(f"TTS URL: {config.get_tts_url()}")
    print(f"STT URL: {config.get_stt_url()}")
    print(f"Moremi URL: {config.get_moremi_url()}")

def test_scheduler_service():
    """Test the SchedulerService class."""
    print("\n=== Testing SchedulerService ===")
    scheduler = SchedulerService()
    
    # Test appointment handling
    conversation = """
    Doctor: I think we should schedule a follow-up in two weeks to check your progress.
    Patient: That sounds good. I'm available on Mondays and Wednesdays.
    Doctor: How about Wednesday, June 15th at 2:00 PM?
    Patient: Yes, that works for me.
    """
    
    print("Testing appointment scheduling...")
    # For testing, we'll disable speaking the response
    response = scheduler.schedule_appointment(conversation, should_speak=False)
    print(f"Scheduler response: {response[:150]}...")
    
    print("Testing appointment summary...")
    summary = scheduler.summarize_appointment(conversation)
    print(f"Summary: {summary}")

def test_diagnosis_service():
    """Test the DifferentialDiagnosisService class."""
    print("\n=== Testing DifferentialDiagnosisService ===")
    diagnosis = DifferentialDiagnosisService()
    
    # Test diagnosis
    symptoms = """
    I've been experiencing severe headaches for the past week, particularly on the right side. 
    The pain is throbbing and gets worse with movement. I've also had some nausea and sensitivity to light.
    """
    
    print("Testing diagnosis...")
    # For testing, we'll disable speaking the response
    response = diagnosis.diagnose(symptoms, should_speak=False)
    print(f"Diagnosis response: {response[:150]}...")
    
    # Create a sample conversation for summary testing
    conversation = f"""
    Patient: {symptoms}
    Doctor: How long have you had these symptoms?
    Patient: About a week now.
    Doctor: Have you taken any medication for it?
    Patient: Just some over-the-counter pain relievers, but they don't help much.
    Doctor: Based on your symptoms, this sounds like it could be a migraine. I recommend...
    """
    
    print("Testing diagnosis summary...")
    summary = diagnosis.summarize_diagnosis(conversation)
    print(f"Summary: {summary}")
    
    print("Testing SOAP note generation...")
    soap_note = diagnosis.generate_soap_note(conversation)
    if isinstance(soap_note, dict) and "error" not in soap_note:
        print("Successfully generated SOAP note")
    else:
        print(f"SOAP note generation issue: {soap_note.get('error', 'Unknown error')}")

def test_medication_service():
    """Test the MedicationReminderService class."""
    print("\n=== Testing MedicationReminderService ===")
    medication = MedicationReminderService()
    
    # Test medication reminder
    patient_info = {
        "patient_name": "John Smith",
        "medication_name": "Lisinopril",
        "medication_dosage": "10mg",
        "medication_frequency": "once daily in the morning"
    }
    
    print("Testing medication reminder...")
    # For testing, we'll disable speaking the response
    response = medication.handle_medication_reminder(patient_info, should_speak=False)
    print(f"Medication reminder response: {response[:150]}...")

def test_speech_assistant():
    """Test the SpeechAssistant class."""
    print("\n=== Testing SpeechAssistant ===")
    assistant = SpeechAssistant()
    
    # Test service switching and text processing
    inputs = [
        "I'd like to schedule an appointment for next Tuesday.",
        "I've been having severe pain in my abdomen for three days.",
        "I need to refill my prescription for blood pressure medication."
    ]
    
    for text in inputs:
        print(f"\nProcessing input: {text}")
        # For testing, we'll disable speaking the response
        response = assistant.process_input(text, should_speak=False)
        print(f"Response: {response[:150]}...")
    
    # Test conversation saving
    assistant.save_conversation("test_conversation.json")
    print("Conversation saved to test_conversation.json")

def test_speech_recognition():
    """Test speech recognition (optional)."""
    print("\n=== Testing Speech Recognition (Press Ctrl+C to skip) ===")
    try:
        assistant = SpeechAssistant()
        print("Please speak after the prompt...")
        text = assistant.recognize_speech()
        print(f"Recognized text: {text}")
        
        if text:
            response = assistant.process_input(text, should_speak=False)
            print(f"Response: {response[:150]}...")
    except KeyboardInterrupt:
        print("\nSpeech recognition test skipped")
    except Exception as e:
        print(f"Error in speech recognition: {e}")

def main():
    """Run all tests."""
    print("=== Testing Unified AI Service ===")
    
    # Basic tests
    test_config_manager()
    
    # Test individual services
    test_scheduler_service()
    test_diagnosis_service()
    test_medication_service()
    
    # Test the integrated assistant
    test_speech_assistant()
    
    # Optionally test speech recognition (can be skipped with Ctrl+C)
    test_speech_recognition()
    
    print("\n=== All tests completed ===")

if __name__ == "__main__":
    main()