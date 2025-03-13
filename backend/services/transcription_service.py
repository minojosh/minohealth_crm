from typing import Optional, Union
import base64
import logging
import numpy as np

logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(self, stt_client):
        self.stt_client = stt_client
        
    async def transcribe(self, audio_data: Union[str, bytes], is_base64: bool = True) -> Optional[str]:
        """
        Transcribe audio data using the STT client
        
        Args:
            audio_data: Audio data as base64 string or bytes
            is_base64: Whether the audio_data is base64 encoded
            
        Returns:
            Transcription text or None if transcription failed
        """
        try:
            audio_bytes = base64.b64decode(audio_data) if is_base64 else audio_data
            transcription = self.stt_client.transcribe_audio(audio_bytes)
            
            if transcription:
                logger.info(f"Transcription completed: {transcription[:50]}...")
                
            # Filter out specific error messages that should be treated as empty transcriptions
            if transcription in ["No speech detected", "No transcription result received"]:
                return None
                
            # Check for error patterns
            if transcription and (
                transcription.startswith("Error") or 
                transcription.startswith("Connection error") or
                transcription.startswith("Server error") or
                transcription.startswith("Processing error")
            ):
                logger.error(f"Transcription service returned error: {transcription}")
                return None
                
            return transcription
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return None
            
    def safe_frombuffer(self, buffer_data, dtype=np.float32):
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
