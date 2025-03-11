import os
import requests
import base64
import sounddevice as sd
import numpy as np
import io
import soundfile as sf
import time
import sys
from IPython import get_ipython, display
import IPython.display
from dotenv import load_dotenv
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTSClient:
    def __init__(self, api_url=None, timeout=15):
        """
        Initialize the TTS client
        
        Args:
            api_url (str): URL of the TTS server
            timeout (int): Timeout for server connection in seconds
        """
        # Load environment variables from .env file
        load_dotenv()

        # Initialize server URL - prioritize provided URL, then env var, then default
        if api_url is not None:
            self.api_url = api_url.rstrip('/')
        else:
            env_url = os.getenv('SPEECH_SERVICE_URL')
            if env_url:
                self.api_url = env_url.rstrip('/')
            else:
                self.api_url = "http://localhost:8000"

        logger.info(f"Using Speech Service URL: {self.api_url}")
        
        # Set timeout for server connection (increased for potentially slow ngrok connections)
        self.connection_timeout = timeout
        self.server_available = self.check_server()
        
        # If server is not available, log but don't fail - we'll retry later
        if not self.server_available:
            logger.warning("Initial server check failed, will retry on first TTS request")

    def check_server(self):
        """
        Check if the server is ready - more robust check that tries multiple endpoints
        
        Returns:
            bool: True if server is ready, False otherwise
        """
        logger.info("Checking if server is ready...")
        
        # Try multiple potential endpoints to increase chances of success
        endpoints = [
            "/", 
            "/health",
            "/generate_speech"  # Just checking if the endpoint exists, not sending data
        ]
        
        for endpoint in endpoints:
            try:
                url = f"{self.api_url}{endpoint}"
                logger.info(f"Testing endpoint: {url}")
                
                response = requests.get(
                    url, 
                    timeout=self.connection_timeout,
                    headers={"Accept": "application/json"}
                )
                
                # Any successful connection is good enough
                if response.status_code < 500:
                    logger.info(f"Server responded with status {response.status_code} from {endpoint}")
                    return True
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to connect to {endpoint}: {str(e)}")
                continue
        
        # If we get here, no endpoint was reachable
        logger.warning(f"Server not available at {self.api_url}")
        return False

    def generate_speech(self, text, speaker="jude"):
        """
        Generate speech from text using the TTS API
        
        Args:
            text (str): Text to convert to speech
            speaker (str): Name of voice to use
            
        Returns:
            tuple: (audio_data, sample_rate) where audio_data is numpy array of audio samples
                  and sample_rate is the audio sample rate in Hz
        """
        # Retry server check if it wasn't available initially
        if not self.server_available:
            logger.info("Server wasn't available initially, retrying check now...")
            self.server_available = self.check_server()
            
        if not self.server_available:
            logger.warning("TTS server is still not available, returning empty audio data")
            return np.array([]).astype(np.float32), 16000
            
        # First try the /generate_speech endpoint (used in ssr.py)
        endpoints = [
            "/generate_speech",
            "/tts"  # Fallback to /tts endpoint
        ]
        
        for endpoint in endpoints:
            try:
                url = f"{self.api_url}{endpoint}"
                logger.info(f"Sending TTS request to: {url}")
                
                response = requests.post(
                    url,
                    json={"text": text, "speaker": speaker},
                    timeout=30  # Longer timeout for speech generation
                )
                
                if response.status_code == 200:
                    logger.info("TTS request successful")
                    data = response.json()
                    
                    # Handle different response formats
                    audio_base64 = data.get("audio_base64") or data.get("audio")
                    sample_rate = data.get("sample_rate", 24000)
                    
                    if audio_base64:
                        # Decode base64 audio
                        audio_bytes = base64.b64decode(audio_base64)
                        
                        try:
                            # Try reading as wav file first
                            audio_buffer = io.BytesIO(audio_bytes)
                            audio_data, _ = sf.read(audio_buffer)
                            return audio_data, sample_rate
                        except Exception:
                            # Fallback to direct interpretation as float32 data
                            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                            return audio_data, sample_rate
                    else:
                        logger.warning("Response contained no audio data")
                else:
                    logger.warning(f"Server error from {endpoint}: {response.status_code}")
                    logger.warning(f"Response: {response.text[:200]}")
            
            except requests.exceptions.Timeout:
                logger.warning(f"Request timed out on {endpoint}. The server took too long to respond.")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error on {endpoint}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error on {endpoint}: {e}")
        
        # If we reach here, all endpoints failed
        logger.error("All TTS endpoints failed")
        return np.array([]).astype(np.float32), 16000

    def get_audio_as_base64(self, text, speaker="jude"):
        """
        Generate speech and return as base64 encoded string ready for frontend use
        
        Args:
            text (str): Text to convert to speech
            speaker (str): Name of voice to use
            
        Returns:
            tuple: (base64_audio, sample_rate) where base64_audio is a base64 encoded string
                  and sample_rate is the audio sample rate in Hz
        """
        audio_data, sample_rate = self.generate_speech(text, speaker)
        if audio_data is not None and len(audio_data) > 0:
            # Convert numpy array to bytes
            audio_bytes = audio_data.tobytes()
            # Convert bytes to base64 string
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            return base64_audio, sample_rate
        return None, sample_rate

    def play_audio(self, audio_data, sample_rate):
        """
        Play audio through local speakers
        
        Args:
            audio_data (numpy.ndarray): Audio data to play
            sample_rate (int): Sample rate in Hz
        """
        if audio_data is not None and len(audio_data) > 0:
            try:
                logger.info("Playing audio through speakers...")
                sd.play(audio_data, sample_rate)
                sd.wait()  # Wait until audio finishes playing
                logger.info("Audio playback complete")
            except Exception as e:
                logger.warning(f"Error playing audio: {e}")
                # Fall back to IPython display if available
                try:
                    logger.info("Trying to display audio with IPython as fallback...")
                    display(IPython.display.Audio(audio_data, rate=sample_rate))
                except:
                    logger.warning("Unable to play or display audio")
    
    def TTS(self, text, play_locally=True, return_data=False):
        """
        Generate speech from text and either play it locally or return the data
        
        Args:
            text (str): Text to convert to speech
            play_locally (bool): Whether to play the audio locally
            return_data (bool): Whether to return the audio data and sample rate
        """
        if not text:
            logger.warning("No text provided for TTS")
            return np.array([]).astype(np.float32), 16000, None if return_data else (None, None, None)

        # Generate audio
        audio_data, sample_rate = self.generate_speech(text)
        
        # Convert to base64 if returning data
        base64_audio = None
        if return_data and len(audio_data) > 0:
            audio_bytes = audio_data.tobytes()
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            logger.info(f"Generated base64 audio data, length: {len(base64_audio)} characters")
        
        # Play locally if requested
        if play_locally and len(audio_data) > 0:
            logger.info("Playing audio...")
            self.play_audio(audio_data, sample_rate)
        
        # Return data if requested
        if return_data:
            return audio_data, sample_rate, base64_audio
        
        return None, None, None

    def get_audio_for_frontend(self, text, speaker="jude"):
        """
        Generate audio specifically formatted for frontend use
        
        Args:
            text (str): Text to convert to speech
            speaker (str): Name of voice to use
            
        Returns:
            dict: Dictionary with audio data ready for WebSocket transmission
                 {
                     "type": "audio",
                     "audio": base64_encoded_string,
                     "sample_rate": sample_rate
                 }
                 Returns None if generation fails
        """
        base64_audio, sample_rate = self.get_audio_as_base64(text, speaker)
        if base64_audio:
            return {
                "type": "audio",
                "audio": base64_audio,
                "sample_rate": sample_rate
            }
        return None


if __name__ == "__main__":
    client = TTSClient()
    client.TTS("Hello, this is a test of the text to speech client.", play_locally=True)
