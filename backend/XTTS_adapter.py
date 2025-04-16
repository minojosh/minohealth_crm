import os
import base64
import numpy as np
import logging
import time
import requests
import json
import io
from pathlib import Path
from dotenv import load_dotenv
import queue

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the updated client but handle the type error
try:
    from .XTTS_client import TTSClient as XTTSClient
except ImportError as e:
    if "typint" in str(e):
        # Try to patch the typing import error
        import importlib.util
        import sys
        
        file_path = os.path.join(os.path.dirname(__file__), "XTTS_client.py")
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read().replace('from typint import', 'from typing import')
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            # Now try to import again
            from .XTTS_client import TTSClient as XTTSClient
        else:
            logger.error(f"Cannot find XTTS_client.py to patch: {file_path}")
            raise
    else:
        raise

class TTSClient:
    """
    Adapter class to provide backward compatibility with the old TTSClient API
    while using the new XTTS implementation under the hood.
    """
    def __init__(self, api_url=None, audio_dir=None):
        """
        Initialize the TTS client adapter
        
        Args:
            api_url (str): URL of the TTS server
            timeout (int): Timeout for server connection in seconds
            audio_dir (str): Directory to save audio files
        """
        # Load environment variables from .env file
        dotenv_path = Path(__file__).parent.parent / '.env'
        load_dotenv(dotenv_path=dotenv_path, encoding='utf-8')


        # Initialize server URL - prioritize provided URL, then env var, then default
        if api_url is not None:
            self.api_url = api_url.rstrip('/')
        else:
            env_url = os.getenv('XTTS_URL') or os.getenv('STT_SERVER_URL')
            if env_url:
                self.api_url = env_url.rstrip('/')
            else:
                self.api_url = "http://localhost:8000"

        logger.info(f"Using XTTS Speech Service URL: {self.api_url}")
        
        # Setup audio directory
        self.audio_dir = audio_dir or "backend/data/audio"
        os.makedirs(self.audio_dir, exist_ok=True)
        logger.info(f"Audio files will be saved to: {self.audio_dir}")
        
        # Set default target sample rate
        self.target_sample_rate = 24000
        

        
        # These are used to store the latest generated audio
        self.last_audio_data = None
        self.last_sample_rate = self.target_sample_rate
        self.last_base64_audio = None
        self.last_file_path = None
        
        # Initialize the new streaming client
        try:
            self.streaming_client = XTTSClient(server_url=self.api_url)
            logger.info("Successfully initialized the streaming TTS client")
        except Exception as e:
            logger.error(f"Failed to initialize streaming client: {e}")
            self.streaming_client = None

    def _direct_request_to_server(self, text, language="en"):
        """
        Make a direct request to the XTTS server and collect audio chunks
        
        This bypasses the XTTS_client and makes a direct request to match
        the server's expected API.
        
        Args:
            text (str): Text to convert to speech
            language (str): Language code
            
        Returns:
            list: List of raw audio chunks
        """
        data = {"text": text, "language": language}
        all_audio_chunks = []
        
        try:
            # Make streaming request to server
            logger.info(f"Sending direct request to TTS server: {text[:30]}...")
            response = requests.post(
                f"{self.api_url}/tts-stream",
                json=data,
                stream=True
            )

            if response.status_code != 200:
                logger.error(f"Server error: {response.status_code} - {response.text}")
                return all_audio_chunks

            # Process streaming response
            buffer = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    buffer += line_str

                    # Try to parse JSON from buffer
                    try:
                        chunk_data = json.loads(buffer)
                        buffer = ""

                        # Check if there's an error
                        if "error" in chunk_data:
                            logger.error(f"Server reported error: {chunk_data['error']}")
                            continue

                        # Decode base64 audio chunk
                        if "chunk" in chunk_data:
                            audio_bytes = base64.b64decode(chunk_data['chunk'])
                            all_audio_chunks.append(audio_bytes)
                    except json.JSONDecodeError:
                        # Incomplete JSON, continue adding to buffer
                        continue
                    
            logger.info(f"Received {len(all_audio_chunks)} audio chunks")
            return all_audio_chunks
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return all_audio_chunks

    def save_audio_file(self, audio_data, sample_rate, file_name=None):
        """
        Save audio data to a file
        
        Args:
            audio_data (numpy.ndarray): Audio data as numpy array
            sample_rate (int): Sample rate of the audio
            file_name (str): Name of the output file (without extension)
            
        Returns:
            str: Path to the saved file
        """
        if audio_data is None or len(audio_data) == 0:
            logger.warning("No audio data to save")
            return None
            
        # Generate filename based on timestamp if none provided
        if not file_name:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            file_name = f"tts_{timestamp}"
            
        # Add wav extension if not present
        if not file_name.endswith('.wav'):
            file_name += '.wav'
            
        # Create full file path
        file_path = os.path.join(self.audio_dir, file_name)
        
        try:
            # Convert float32 back to int16 for saving
            audio_int16 = (audio_data * 32768.0).astype(np.int16)
            
            # Save as WAV file
            import wave
            with wave.open(file_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
                
            logger.info(f"Audio saved to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            return None

    def TTS(self, text, save_audio=True, file_name=None):
        """
        Generate speech from text with compatibility with the old API
        
        Args:
            text (str): Text to convert to speech
            play_locally (bool): Whether to play the audio locally (handled by streaming client if enabled)
            return_data (bool): Whether to return the audio data
            save_audio (bool): Whether to save the audio to a file
            file_name (str): Name for the saved audio file
            
        Returns:
            tuple: (audio_data, sample_rate, base64_audio, file_path)
        """
        if not text:
            logger.warning("No text provided for TTS")
            return np.array([]).astype(np.float32), self.target_sample_rate, None, None
        
        all_audio_chunks = []
        # Try to use streaming client if available
        if self.streaming_client:
            try:
                # Queue text for streaming playback
                logger.info(f"Using streaming client for text: {text[:30]}...")
                for audio_chunk in self.streaming_client.stream_text(text):
                    if audio_chunk:
                        logger.info(f"Received audio chunk of size: {len(audio_chunk)} bytes")
                        all_audio_chunks.append(audio_chunk)
                        # Encode the raw audio bytes directly
                        base64_audio = base64.b64encode(audio_chunk).decode('utf-8')
                        yield base64_audio
                        

            except Exception as e:
                logger.error(f"Error using streaming client: {e}")
                all_audio_chunks = []
        
        else:
            # Collect audio chunks from server (this is needed even if streaming, to return the data)
            all_audio_chunks = self._direct_request_to_server(text)
    
        
        if not all_audio_chunks:
            logger.warning("No audio chunks received from server")
            return np.array([]).astype(np.float32), self.target_sample_rate, None, None
        
        # Combine all audio chunks
        audio_bytes = b''.join(all_audio_chunks)
        logger.info(f"Combined audio bytes length: {len(audio_bytes)} bytes")
        
        # Verify we have valid audio bytes
        if len(audio_bytes) < 4:
            logger.error("Audio bytes are too small to be valid audio")
            return np.array([]).astype(np.float32), self.target_sample_rate, None, None
            
        # Convert to numpy array for processing
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Log audio statistics for debugging
        if len(audio_data) > 0:
            min_val = np.min(audio_data)
            max_val = np.max(audio_data)
            mean_val = np.mean(audio_data)
            logger.info(f"Audio data statistics - min: {min_val:.4f}, max: {max_val:.4f}, mean: {mean_val:.4f}, length: {len(audio_data)} samples")
            
          
            file_path = self.save_audio_file(audio_data, self.target_sample_rate, file_name)
        
        # Convert to base64 for frontend use - make sure we use the raw int16 bytes, not the float32 data
        base64_audio = None
        try:
            # Encode the raw audio bytes directly
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            logger.info(f"Encoded base64 audio length: {len(base64_audio)} characters")
            
            # Verify the base64 encoding was successful
            if not base64_audio or len(base64_audio) == 0:
                logger.error("Base64 encoding failed - empty string returned")
            
            # Test validity of base64 string (should not throw exception)
            test_decode = base64.b64decode(base64_audio.encode('utf-8'))
            if len(test_decode) != len(audio_bytes):
                logger.warning(f"Base64 decode test returned different length: expected {len(audio_bytes)}, got {len(test_decode)}")
                
        except Exception as e:
            logger.error(f"Error encoding audio to base64: {e}")
            base64_audio = None
        
        # Add a None for the file_path parameter (expected by calling code)
        file_path = None
            
        return audio_data, self.target_sample_rate, base64_audio, file_path
       
        
    def get_audio_for_frontend(self, text, speaker="default", save_audio=True, file_name=None):
        """
        Generate audio formatted for frontend consumption
        
        Args:
            text (str): Text to convert to speech
            speaker (str): Speaker voice to use (not used in XTTS)
            save_audio (bool): Whether to save the audio to a file
            file_name (str): Name for the saved audio file
            
        Returns:
            dict: Dictionary with audio data and metadata
        """
        audio_data, sample_rate, base64_audio, file_path = self.TTS(
            text, 
            play_locally=False, 
            return_data=True, 
            save_audio=save_audio, 
            file_name=file_name
        )
        
        if base64_audio:
            result = {
                "type": "audio",
                "audio": base64_audio,
                "sample_rate": sample_rate
            }
            
            if file_path:
                result["file_path"] = file_path
                
            return result
        else:
            return {"type": "error", "message": "Failed to generate audio"}

    def get_audio_as_base64(self, text, speaker="default"):
        """
        Generate audio and return as base64 string
        
        Args:
            text (str): Text to convert to speech
            speaker (str): Speaker voice to use (not used in XTTS)
            
        Returns:
            str: Base64 encoded audio data
        """
        _, _, base64_audio, _ = self.TTS(text, play_locally=False, return_data=True, save_audio=False)
        return base64_audio
        
    def stream_text(self, text):
        """
        Stream text through the streaming client if available
        
        Args:
            text (str): Text to convert to speech and stream
        """
        if self.streaming_client:
            try:
                self.TTS(text)
                return True
            except Exception as e:
                logger.error(f"Error streaming text: {e}")
                return False
        else:
            logger.warning("Streaming client not available, using synchronous TTS instead")
            self.TTS(text, play_locally=True)
            return False
        
    
            
    def wait_for_completion(self, timeout=1):
        """
        Wait for streaming audio to complete
        
        Args:
            timeout (int): Maximum time to wait in seconds
        """
        if self.streaming_client:
            try:
                self.streaming_client.wait_for_completion(timeout)
            except Exception as e:
                logger.error(f"Error waiting for streaming completion: {e}")
                
    def stop(self):
        """
        Stop any ongoing streaming audio
        """
        if self.streaming_client:
            try:
                self.streaming_client.stop()
            except Exception as e:
                logger.error(f"Error stopping streaming client: {e}")

    def generate_audio(self, text, play_locally=False, return_data=True, save_audio=False):
        """Generate audio chunks from text using the streaming client.
        
        Args:
            text (str): Text to convert to speech
            play_locally (bool): Whether to play audio locally
            return_data (bool): Whether to return audio data
            save_audio (bool): Whether to save the audio
            
        Yields:
            tuple: (audio_bytes, sample_rate, base64_audio) for each chunk
        """
        if not text:
            logger.warning("No text provided for TTS")
            return
        
        try:
            # Use streaming client if available
            if self.streaming_client:
                logger.info(f"Using streaming client for text: {text[:30]}...")
                self.streaming_client.stream_text(text)
                
                # Process chunks as they come in
                for chunk in self.streaming_client._send_to_tts(text):
                    if chunk:
                        # Convert to base64
                        base64_audio = base64.b64encode(chunk).decode('utf-8')
                        yield chunk, self.target_sample_rate, base64_audio
                        
            else:
                logger.warning("Streaming client not available")
                
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return