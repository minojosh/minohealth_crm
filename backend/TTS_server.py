import os
import sys
import base64
import logging
import nest_asyncio
import soundfile as sf
import numpy as np
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from pathlib import Path
from yarngpt.audiotokenizer import AudioTokenizer
from transformers import AutoModelForCausalLM
import torch
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tts_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define request model
class GenerateSpeechRequest(BaseModel):
    text: str
    speaker: str = "jude"  # Default speaker

# Apply nest_asyncio patch to allow nested event loops
nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI(title="MinoHealth TTS Server")

@app.get("/")
async def root():
    """Root endpoint to check if server is running"""
    return {"status": "running", "service": "MinoHealth TTS Server"}

# Variables to store model and tokenizer
model = None
audio_tokenizer = None

def initialize_model():
    """Initialize the TTS model and tokenizer"""
    global model, audio_tokenizer
    
    if model is not None and audio_tokenizer is not None:
        return
    
    try:
        # Get the base directory for paths
        base_dir = Path(__file__).parent.resolve()
        logger.info(f"Base directory: {base_dir}")
        
        # Define paths relative to the current file
        # You'll need to download these files and adjust these paths
        tokenizer_path = "saheedniyi/YarnGPT2b"
        wav_tokenizer_config_path = str(base_dir / "models" / "wavtokenizer_config.yaml")
        wav_tokenizer_model_path = str(base_dir / "models" / "wavtokenizer_model.ckpt")
        
        logger.info(f"Loading audio tokenizer from {tokenizer_path}")
        logger.info(f"Config path: {wav_tokenizer_config_path}")
        logger.info(f"Model path: {wav_tokenizer_model_path}")
        
        # Check if files exist
        if not os.path.exists(wav_tokenizer_config_path):
            logger.error(f"Config file not found: {wav_tokenizer_config_path}")
            raise FileNotFoundError(f"Config file not found: {wav_tokenizer_config_path}")
        
        if not os.path.exists(wav_tokenizer_model_path):
            logger.error(f"Model file not found: {wav_tokenizer_model_path}")
            raise FileNotFoundError(f"Model file not found: {wav_tokenizer_model_path}")
        
        # Create the AudioTokenizer object
        audio_tokenizer = AudioTokenizer(tokenizer_path, wav_tokenizer_model_path, wav_tokenizer_config_path)
        
        # Load the model weights
        logger.info(f"Loading model from {tokenizer_path}")
        model = AutoModelForCausalLM.from_pretrained(tokenizer_path, torch_dtype=torch.float16)
        
        # Move model to appropriate device (CPU or GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"Model loaded and moved to device: {device}")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

@app.post("/generate_speech")
async def generate_speech(request: GenerateSpeechRequest):
    """Generate speech from text and return as base64 encoded audio"""
    try:
        # Initialize model if not already done
        if model is None or audio_tokenizer is None:
            initialize_model()
        
        text = request.text
        speaker = request.speaker
        logger.info(f"Generating speech for text: '{text}' with speaker: {speaker}")
        
        # Check if text is provided
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        # Generate audio - this will depend on your specific model implementation
        # Assuming audio_tokenizer.generate_audio returns a numpy array
        audio_data = audio_tokenizer.generate_audio(text, speaker=speaker)
        
        # Get sample rate from audio tokenizer
        sample_rate = audio_tokenizer.sample_rate  # Adjust based on your tokenizer's API
        
        # Convert audio data to WAV file in memory
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, sample_rate, format="WAV")
        wav_buffer.seek(0)
        
        # Convert to base64
        base64_audio = base64.b64encode(wav_buffer.read()).decode("utf-8")
        
        return {
            "audio_base64": base64_audio,
            "sample_rate": sample_rate,
            "text": text,
            "speaker": speaker
        }
    
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate speech: {str(e)}"
        )

def start():
    """Start the FastAPI server"""
    try:
        # Attempt to initialize the model at startup
        initialize_model()
    except Exception as e:
        logger.error(f"Failed to initialize model at startup: {str(e)}")
        print(f"Failed to initialize model: {str(e)}")
        print("Server will attempt to initialize model on first request")
    
    # Get port from environment variable or use default
    port = int(os.environ.get("TTS_SERVER_PORT", 8001))
    print(f"Starting TTS server on port {port}")
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    start()