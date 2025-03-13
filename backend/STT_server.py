!pip install -q git+https://github.com/openai/whisper.git
!pip install -q pyngrok
!pip install nest_asyncio
!pip install aiohttp
!pip install fastapi uvicorn
import asyncio
import numpy as np
import torch
import whisper
import json
import base64
import logging
import time
import uuid
import nest_asyncio
nest_asyncio.apply()
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from aiohttp import web
from pyngrok import ngrok
from collections import deque
from google.colab import userdata

ngrok.set_auth_token("2tM9McmNWocK45xUtJc9SuzrlCM_7PnNAwmaH3jANfhaMnt5F")  # Add your token for better stability
# Clear any previous handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create a dedicated logger
logger = logging.getLogger("whisper_app")
logger.setLevel(logging.INFO)
logger.handlers = []  # Ensure no existing handlers

# Create handlers - just ONE console handler
file_handler = logging.FileHandler("whisper_app.log", mode='w')
console_handler = logging.StreamHandler()  # This is the only console output now

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Suppress other loggers (like uvicorn access logs)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Initialize Whisper model
logger.info("Loading Whisper model...")
torch.cuda.empty_cache()

# Load model with specific compute type for better efficiency
model = whisper.load_model("large")

class AudioProcessor:
    def __init__(self, min_samples=16000 * 10, min_confidence_threshold=0.6):
        self.model = model  # Assuming this is defined elsewhere
        self.buffer = np.array([], dtype=np.float32)
        self.recent_transcriptions = []
        self.current_sentence = ""
        self.min_samples = min_samples
        self.min_confidence_threshold = min_confidence_threshold
        self.last_chunk = ""
        self.accumulated_buffer = np.array([], dtype=np.float32)
        self.accumulated_transcription = ""
        self.accumulated_transcriptions = []
        self.temp_buffer = np.array([], dtype=np.float32)
        self.context_prompt = ""
        self.clients = {}  # Track clients by IP address

    def add_audio(self, audio_array, client_id):
        self.buffer = np.concatenate([self.buffer, audio_array])
        self.accumulated_buffer = np.concatenate([self.accumulated_buffer, audio_array])
        self.temp_buffer = np.concatenate([self.temp_buffer, audio_array])
        logger.debug(f"Client {client_id}: Buffer size after adding audio: {len(self.buffer)} samples")

    def should_process(self):
        return len(self.buffer) >= self.min_samples

    def get_audio(self):
        audio = self.buffer.copy()
        self.buffer = np.array([], dtype=np.float32)
        logger.debug(f"Retrieved {len(audio)} samples for processing, buffer cleared")
        return audio

    def get_temp_audio(self):
        audio = self.temp_buffer.copy()
        self.temp_buffer = np.array([], dtype=np.float32)
        logger.debug(f"Retrieved {len(audio)} samples from temp buffer")
        return audio

    def clear_context(self):
        self.recent_transcriptions = []
        self.last_chunk = ""
        self.context_prompt = ""  # Also clear the context prompt
        logger.info("Clearing context")

    def update_context(self, transcription):
        if not transcription:
            return
            
        # Clean the transcription text
        transcription = transcription.strip()
        if not transcription:
            return
            
        # Add to context with proper spacing
        if self.context_prompt:
            self.context_prompt = f"{self.context_prompt} {transcription}".strip()
        else:
            self.context_prompt = transcription

        # Limit context to last 500 words to prevent memory issues
        words = self.context_prompt.split()
        if len(words) > 500:
            self.context_prompt = " ".join(words[-500:])

        logger.debug(f"Context updated, now contains {len(words)} words")
        
        # Also store in recent transcriptions for alternative access patterns
        self.recent_transcriptions.append(transcription)

        # Keep only the last 10 transcriptions
        if len(self.recent_transcriptions) > 10:
            self.recent_transcriptions = self.recent_transcriptions[-10:]

    def get_context(self):
        return self.context_prompt

    def register_client(self, client_id):
        """Register a client with timestamp and session tracking"""
        if client_id not in self.clients:
            logger.info(f"New client registered: {client_id}")
            self.clients[client_id] = {
                "first_seen": asyncio.get_event_loop().time(),
                "last_seen": asyncio.get_event_loop().time(),
                "request_count": 1,
                "total_audio_samples": 0
            }
        else:
            self.clients[client_id]["last_seen"] = asyncio.get_event_loop().time()
            self.clients[client_id]["request_count"] += 1
            logger.debug(f"Client {client_id}: Request #{self.clients[client_id]['request_count']}")

    def update_client_stats(self, client_id, samples_count):
        """Update client statistics"""
        if client_id in self.clients:
            self.clients[client_id]["total_audio_samples"] += samples_count
            logger.debug(f"Client {client_id}: Total samples received: {self.clients[client_id]['total_audio_samples']}")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processor at startup
@app.on_event("startup")
async def startup_event():
    app.state.processor = AudioProcessor()

    # Start ngrok tunnel
    PORT = 8000
    logger.info("Starting ngrok tunnel...")
    public_url = ngrok.connect(PORT, "http")
    ngrok_url = public_url.public_url
    app.state.ngrok_url = ngrok_url
    app.state.ngrok_tunnel = public_url
    logger.info(f"ngrok tunnel established at: {ngrok_url}")
    print(f"ngrok tunnel established at: {ngrok_url}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down server and closing ngrok tunnel...")
    if hasattr(app.state, 'ngrok_tunnel'):
        ngrok.disconnect(app.state.ngrok_tunnel)
    logger.info("Server shutdown complete.")

@app.get("/")
async def root():
    return {"status": "Server is running", "api_url": f"{app.state.ngrok_url}/transcribe"}

@app.post("/transcribe")
async def transcribe(request: Request):
    processor = app.state.processor
    client_id = f"{request.client.host}:{request.client.port}"

    # Register client activity
    processor.register_client(client_id)

    try:
        data = await request.json()
        logger.debug(f"Client {client_id}: Request type: {data.get('command', 'audio upload')}")

        if 'command' in data and data['command'] == 'finish':
            logger.info(f"Client {client_id}: Finish command received")
            # Process any remaining audio
            final_audio = processor.get_temp_audio()
            if len(final_audio) > 0:
                audio_duration = len(final_audio)/16000
                logger.info(f"Client {client_id}: Processing final chunk of {audio_duration:.2f} seconds")
                result = model.transcribe(
                    final_audio,
                    language="en",
                    task="transcribe",
                    initial_prompt=processor.get_context(),
                    temperature=0.2,
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6
                )
                transcription = result["text"].strip()
                if transcription:
                    processor.update_context(transcription)
                    logger.info(f"Client {client_id}: Final transcription: '{transcription}'")
                    processor.accumulated_transcriptions.append(transcription)
                    return {
                        "transcription": transcription,
                        "is_final": True,
                        "type": "final"
                    }
            processor.clear_context()
            logger.info(f"Client {client_id}: Session complete. Total audio: {processor.clients[client_id]['total_audio_samples']/16000:.2f} seconds")
            return {"status": "ok"}

        if 'audio' in data:
            audio_data = base64.b64decode(data['audio'])
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            level = np.abs(audio_array).mean()  # Calculate audio level
            audio_duration = len(audio_array)/16000

            # Update client stats
            processor.update_client_stats(client_id, len(audio_array))

            # Log audio information with emphasis on duration and quality
            logger.info(f"Client {client_id}: Received {audio_duration:.2f}s of audio, level: {level:.6f}")

            if level < 0.001:
                pass
            else:
                processor.add_audio(audio_array, client_id)

            if processor.should_process():
                audio_to_process = processor.get_audio()
                process_duration = len(audio_to_process)/16000
                logger.info(f"Client {client_id}: Processing {process_duration:.2f}s of audio")

                result = model.transcribe(
                    audio_to_process,
                    language="en",
                    task="transcribe",
                    initial_prompt=processor.get_context(),
                    temperature=0.2,
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6
                )

                transcription = result["text"].strip()
                if transcription:
                    logger.info(f"Client {client_id}: Transcription: '{transcription}'")
                    processor.update_context(transcription)
                    processor.accumulated_transcriptions.append(transcription)
                    processor.temp_buffer = np.array([], dtype=np.float32)
                    return {
                        "transcription": transcription,
                        "is_final": False,
                        "type": "not-final"
                    }
                else:
                    logger.info(f"Client {client_id}: No speech detected in {process_duration:.2f}s audio chunk")

            return {
                "status": "ok",
                "buffer_size": len(processor.buffer),
                "session_info": {
                    "request_count": processor.clients[client_id]["request_count"],
                    "total_seconds": processor.clients[client_id]["total_audio_samples"]/16000
                }
            }

    except Exception as e:
        logger.error(f"Client {client_id}: Error processing request: {str(e)}", exc_info=True)
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Starting audio transcription server with FastAPI")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")  # Set uvicorn log level to warning to suppress HTTP logs
    except KeyboardInterrupt:
        logger.info("\nShutdown requested... closing server")
    finally:
        logger.info("Server stopped.")