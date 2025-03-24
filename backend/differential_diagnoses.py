"""
Moremi Speech Assistant - Production Version
A speech-based medical assistant that uses Azure Cognitive Services for speech-to-text
and text-to-speech functionality, integrated with the Moremi AI model.
"""

import os
import json
import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

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

@dataclass
class Message:
    """Represents a conversation message between user and assistant."""
    user_input: str
    response: Optional[str] = None

    def add_response(self, response: str) -> None:
        """Add assistant's response to the message."""
        self.response = response

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return asdict(self)

class SpeechConfig:
    """Manages speech configuration settings."""
    
    def __init__(self):
        """Initialize speech configuration with environment variables."""
        load_dotenv()
        self.speech_key = os.getenv("SPEECH_KEY")
        self.service_region = os.getenv("SERVICE_REGION")
        self.voice_name = 'en-US-CoraMultilingualNeural'
        self.recognition_language = "en-US"
        
        if not self.speech_key or not self.service_region:
            raise ValueError("Missing required environment variables: SPEECH_KEY or SERVICE_REGION")

    def get_speech_config(self) -> speechsdk.SpeechConfig:
        """Create and return Azure speech configuration."""
        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key, 
            region=self.service_region
        )
        speech_config.speech_synthesis_voice_name = self.voice_name
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceResponse_RequestSentenceBoundary, 
            "true"
        )
        return speech_config

class SpeechAssistant:
    """Main speech assistant class that handles speech recognition and synthesis."""
    
    SILENCE_TIMEOUT = 3  # seconds
    
    def __init__(self):
        """Initialize the speech assistant with necessary configurations."""
        self.config = SpeechConfig()
        self.speech_config = self.config.get_speech_config()
        self.synthesizer = self._setup_synthesizer()
        self.messages: List[Message] = []
        self.api = os.getenv("API_URL")
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load system prompts from configuration file."""
        try:
            with open('./prompt.json', 'r') as file:
                schema = json.load(file)
            self.system_prompt = schema.get("systemprompt", "")
            self.summary_system_prompt = schema.get("summary_systemprompt", "")
        except FileNotFoundError:
            logger.warning("prompt.json not found. Using default empty prompts.")
            self.system_prompt = ""
            self.summary_system_prompt = ""

    def _setup_synthesizer(self) -> speechsdk.SpeechSynthesizer:
        """Set up and configure the speech synthesizer."""
        stream_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config, 
            audio_config=stream_config
        )
        synthesizer.synthesizing.connect(self._stream_audio_callback)
        return synthesizer

    @staticmethod
    def _stream_audio_callback(evt: speechsdk.SpeechSynthesisEventArgs) -> None:
        """Handle streaming audio data."""
        if evt.result.reason == speechsdk.ResultReason.SynthesizingAudio:
            pass  # Process audio chunk if needed

    def recognize_speech(self) -> str:
        """Record and transcribe speech from microphone."""
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, 
            audio_config=audio_config
        )

        complete_transcription = []
        done = False
        last_speech_time = time.time()

        def handle_result(evt):
            nonlocal last_speech_time
            last_speech_time = time.time()
            print('\r' + ' ' * 100, end='\r')
            complete_transcription.append(evt.result.text)

        def handle_interim_result(evt):
            nonlocal last_speech_time
            last_speech_time = time.time()
            print(f'\rRecognizing: {evt.result.text}', end='', flush=True)

        def handle_cancellation(evt):
            nonlocal done
            if evt.reason == speechsdk.CancellationReason.Error:
                logger.error(f'Speech recognition error: {evt.error_details} (Code: {evt.error_code})')
            done = True

        recognizer.recognizing.connect(handle_interim_result)
        recognizer.recognized.connect(handle_result)
        recognizer.canceled.connect(handle_cancellation)

        recognizer.start_continuous_recognition()
        logger.info('Listening for speech input...')

        try:
            while not done:
                time.sleep(.5)
                if time.time() - last_speech_time > self.SILENCE_TIMEOUT:
                    logger.info("Speech input complete")
                    done = True
        except KeyboardInterrupt:
            logger.info("Speech recognition interrupted by user")
            done = True
        finally:
            recognizer.stop_continuous_recognition()

        return ' '.join(complete_transcription)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_moremi_response(self, messages: List[Message], system_prompt: Optional[str] = None) -> str:
        """Get response from Moremi API with retry logic."""
        data = {
            "query": [msg.to_dict() for msg in messages],
            "temperature": 1.0,
            "max_new_token": 500
        }
        if system_prompt:
            data["systemPrompt"] = system_prompt

        try:
            response = requests.post(self.api, json=data)
            response.raise_for_status()
            return json.loads(response.text.strip())
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Moremi API: {str(e)}")
            raise

    def synthesize_speech(self, text: str) -> None:
        """Synthesize text to speech."""
        try:
            logger.info("Synthesizing speech...")
            result = self.synthesizer.speak_text_async(text).get()
            
            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.error("Speech synthesis failed")
                raise Exception("Speech synthesis failed")
                
        except Exception as e:
            logger.error(f"Error in speech synthesis: {str(e)}")
            raise

    def save_conversation(self) -> None:
        """Save the conversation history to a file."""
        try:
            with open('conversation.txt', 'w') as f:
                json.dump([msg.to_dict() for msg in self.messages], f, indent=4)
            logger.info("Conversation saved successfully")
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")

    def run(self) -> None:
        """Run the main conversation loop."""
        # Initial greeting
        initial_response = 'Hello! Welcome to Pembroke Hospital. How can I help you today?'
        self.messages.append(Message(user_input='Hi', response=initial_response))
        self.synthesize_speech(initial_response)
        
        logger.info("Starting conversation")
        print("Starting conversation. Press Ctrl+C to exit.")

        while True:
            try:
                # Get speech input
                user_input = self.recognize_speech()
                if not user_input.strip():
                    logger.warning("Empty input received. Please say something.")
                   
                    continue

                logger.info(f"User input: {user_input}")
                
                # Get Moremi response
                self.messages.append(Message(user_input=user_input))
                response = self.get_moremi_response(self.messages, self.system_prompt)
                self.messages[-1].add_response(response)
                
                logger.info(f"Moremi response: {response}")
                print(f'\nMoremi: {response}')

                # Synthesize response
                self.synthesize_speech(response)
                print("\nReady for next input...")

            except KeyboardInterrupt:
                logger.info("Conversation ended by user")
                print('\nEnding conversation...')
                
                # Get conversation summary
                try:
                    summary = self.get_moremi_response(self.messages, self.summary_system_prompt)
                    print(f'\nSummary: {summary}')
                except Exception as e:
                    logger.error(f"Failed to generate summary: {str(e)}")
                
                self.save_conversation()
                break
                
            except Exception as e:
                logger.error(f"Error in conversation loop: {str(e)}")
                error_response = 'Sorry. The system is currently unavailable.'
                try:
                    self.synthesize_speech(error_response)
                except:
                    pass
                break

def main():
    """Main entry point of the application."""
    try:
        assistant = SpeechAssistant()
        assistant.run()
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}")
        print("An error occurred. Please check the logs for details.")

if __name__ == "__main__":
    main()
