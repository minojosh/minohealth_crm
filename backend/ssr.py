!pip install -q git+https://github.com/openai/whisper.git
!pip install -q pyngrok
!pip install nest_asyncio
!pip install aiohttp
!pip install fastapi uvicorn

!sudo apt-get install portaudio19-dev
!pip install trafilatura uroman sounddevice outetts --upgrade
!git clone https://github.com/saheedniyi02/yarngpt.git

!wget https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml
!wget https://huggingface.co/novateur/WavTokenizer-large-speech-75token/resolve/main/wavtokenizer_large_speech_320_24k.ckpt
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
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
from aiohttp import web
from pyngrok import ngrok
from collections import deque
from google.colab import userdata
import os
import re
import json
import inflect
import random
import uroman
import sounddevice
import soundfile as sf
import io
import torchaudio
import IPython
from transformers import AutoModelForCausalLM, AutoTokenizer
from outetts.wav_tokenizer.decoder import WavTokenizer
from yarngpt.audiotokenizer import AudioTokenizerV2
import trafilatura
import requests
from pydantic import BaseModel,Field


# ngrok.set_auth_token
auth_token = userdata.get('auth_token')
ngrok.set_auth_token(auth_token)


# Apply nest_asyncio patch for Colab compatibility
nest_asyncio.apply()

class UnifiedSpeechConfig:
    """Configuration for both STT and TTS models"""
    def __init__(self):
        self.WHISPER_MODEL = "large"
        self.TOKENIZER_PATH = "saheedniyi/YarnGPT2b"
        self.WAV_TOKENIZER_CONFIG = "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        self.WAV_TOKENIZER_MODEL = "wavtokenizer_large_speech_320_24k.ckpt"
        self.MIN_AUDIO_SAMPLES = 16000 * 10  # 10 seconds at 16kHz
        self.MIN_CONFIDENCE = 0.6
        self.PORT = 8000
        self.HOST = "0.0.0.0"
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UnifiedLogger:
    """Centralized logging configuration"""
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler
        file_handler = logging.FileHandler(f"{name}.log")
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

class ModelManager:
    """Manages STT and TTS models with efficient resource usage"""
    def __init__(self, config: UnifiedSpeechConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.stt_model = None
        self.tts_model = None
        self.tokenizer_path = self.config.TOKENIZER_PATH
        self.wav_tokenizer_config = self.config.WAV_TOKENIZER_CONFIG
        self.wav_tokenizer_model = self.config.WAV_TOKENIZER_MODEL
        self.tokenizer = None
        self.audio_tokenizer = None  # Fixed: Initialize in initialize_models instead
        self.cache_dir = "model_cache"
        self._models_initialized = False
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cached_path(self, model_name):
        return os.path.join(self.cache_dir, f"{model_name}.pt")

    def _save_model(self, model, model_name):
        """Save model weights to disk, but don't duplicate in memory"""
        cache_path = self._get_cached_path(model_name)
        if not os.path.exists(cache_path):
            self.logger.info(f"Saving {model_name} to {cache_path}")
            torch.save(model.state_dict(), cache_path)
        else:
            self.logger.info(f"{model_name} already exists at {cache_path}")

    def _load_cached_model(self, model_name):
        """Check if cached model exists"""
        cache_path = self._get_cached_path(model_name)
        if os.path.exists(cache_path):
            self.logger.info(f"Found cached {model_name} at {cache_path}")
            return cache_path
        return None

    async def initialize_models(self):
        """Initialize both STT and TTS models with resource efficiency"""
        if self._models_initialized:
            self.logger.info("Models already initialized, skipping...")
            return True

        try:
            if torch.cuda.is_available():
                self.logger.info("Cleaning GPU memory before loading models...")
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
                self.logger.info(f"Initial GPU memory usage: {initial_memory / 1024**2:.2f} MB")

            # Initialize Whisper model for STT
            whisper_cache = self._load_cached_model("whisper")
            self.logger.info(f"Loading Whisper model ({self.config.WHISPER_MODEL})...")
            self.stt_model = whisper.load_model(self.config.WHISPER_MODEL, device=self.config.DEVICE)

            if whisper_cache:
                # Just log that we found the cache, but no need to reload weights if we're already loading the model
                self.logger.info(f"Whisper model cache found: {whisper_cache}")
            else:
                # Save model for future use
                self._save_model(self.stt_model, "whisper")

            # Initialize tokenizer first (shared between TTS model and audio tokenizer)
            self.logger.info(f"Loading tokenizer from {self.config.TOKENIZER_PATH}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.TOKENIZER_PATH)

            # Initialize TTS model
            self.logger.info("Loading TTS model...")
            dtype = torch.float16 if self.config.DEVICE.type == "cuda" else torch.float32
            self.tts_model = AutoModelForCausalLM.from_pretrained(
                self.config.TOKENIZER_PATH,
                torch_dtype=dtype,
                device_map=self.config.DEVICE
            )

            # Initialize audio tokenizer
            self.logger.info("Loading audio tokenizer...")
            self.audio_tokenizer = AudioTokenizerV2(
                self.config.TOKENIZER_PATH,
                self.config.WAV_TOKENIZER_MODEL,
                self.config.WAV_TOKENIZER_CONFIG
            )

            # Log memory usage after loading models
            if torch.cuda.is_available():
                after_loading = torch.cuda.memory_allocated()
                memory_used = after_loading - initial_memory
                self.logger.info(f"GPU memory used for models: {memory_used / 1024**2:.2f} MB")

            self._models_initialized = True
            self.logger.info("All models initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}", exc_info=True)
            raise

class AudioProcessor:
    """Handles audio processing for both STT and TTS"""
    def __init__(self, config: UnifiedSpeechConfig, model_manager: ModelManager, logger: logging.Logger):
        self.config = config
        self.model_manager = model_manager
        self.logger = logger
        self.buffer = np.array([], dtype=np.float32)
        self.temp_buffer = np.array([], dtype=np.float32)
        self.accumulated_buffer = np.array([], dtype=np.float32)
        self.context_prompt = ""
        self.accumulated_transcriptions = []
        self.clients = {}

    def add_audio(self, audio_array, client_id):
        self.buffer = np.concatenate([self.buffer, audio_array])
        self.accumulated_buffer = np.concatenate([self.accumulated_buffer, audio_array])
        self.temp_buffer = np.concatenate([self.temp_buffer, audio_array])
        self.logger.debug(f"Client {client_id}: Buffer size after adding audio: {len(self.buffer)} samples")

    def should_process(self):
        return len(self.buffer) >= self.config.MIN_AUDIO_SAMPLES

    def get_audio(self):
        audio = self.buffer.copy()
        self.buffer = np.array([], dtype=np.float32)
        self.logger.debug(f"Retrieved {len(audio)} samples for processing, buffer cleared")
        return audio

    def get_temp_audio(self):
        audio = self.temp_buffer.copy()
        self.temp_buffer = np.array([], dtype=np.float32)
        self.logger.debug(f"Retrieved {len(audio)} samples from temp buffer")
        return audio

    def clear_context(self):
        self.context_prompt = ""
        self.logger.info("Clearing context")

    def update_context(self, transcription):
        if transcription:
            self.context_prompt = f"{self.context_prompt} {transcription}".strip()
            words = self.context_prompt.split()
            if len(words) > 500:
                self.context_prompt = " ".join(words[-500:])
            self.logger.debug(f"Context updated, now contains {len(words)} words")

    def get_context(self):
        return self.context_prompt

    def register_client(self, client_id):
        if client_id not in self.clients:
            self.logger.info(f"New client registered: {client_id}")
            self.clients[client_id] = {
                "first_seen": asyncio.get_event_loop().time(),
                "last_seen": asyncio.get_event_loop().time(),
                "request_count": 1,
                "total_audio_samples": 0
            }
        else:
            self.clients[client_id]["last_seen"] = asyncio.get_event_loop().time()
            self.clients[client_id]["request_count"] += 1
            self.logger.debug(f"Client {client_id}: Request #{self.clients[client_id]['request_count']}")

    def update_client_stats(self, client_id, samples_count):
        if client_id in self.clients:
            self.clients[client_id]["total_audio_samples"] += samples_count
            self.logger.debug(f"Client {client_id}: Total samples received: {self.clients[client_id]['total_audio_samples']}")

    def generate_speech(self, text: str, speaker: str = "default") -> Dict[str, Any]:
        """Generate speech using TTS - NOT an async function"""
        try:
            self.logger.info(f"Generating speech for text: '{text}' with speaker: {speaker}")

            # Create a prompt using the correct YarnGPT API
            prompt = self.model_manager.audio_tokenizer.create_prompt(
                text,
                lang="english",  # You might want to make this configurable
                speaker_name=speaker
            )

            # Tokenize the prompt
            input_ids = self.model_manager.audio_tokenizer.tokenize_prompt(prompt)

            # Generate output using the model
            with torch.no_grad():
                output = self.model_manager.tts_model.generate(
                    input_ids=input_ids,
                    temperature=0.1,
                    repetition_penalty=1.1,
                    max_length=4000
                )

            # Extract codes and convert to audio
            codes = self.model_manager.audio_tokenizer.get_codes(output)
            audio_data = self.model_manager.audio_tokenizer.get_audio(codes)

            sample_rate = 24000  # Default sample rate for YarnGPT models

            # Save to temporary file first (torchaudio works better with files than BytesIO)
            temp_wav_path = f"/tmp/temp_speech_{uuid.uuid4()}.wav"
            torchaudio.save(temp_wav_path, audio_data, sample_rate)

            # Read the file and convert to base64
            with open(temp_wav_path, 'rb') as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")

            # Clean up temporary file
            os.remove(temp_wav_path)

            self.logger.info(f"Successfully generated speech ({audio_data.shape[1]} samples)")

            return {
                "audio_base64": audio_base64,
                "sample_rate": sample_rate,
                "text": text,
                "speaker": speaker
            }
        except Exception as e:
            self.logger.error(f"Speech generation error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")

# FastAPI Models
class TranscribeRequest(BaseModel):
    audio: str = Field(..., description="Base64 encoded audio data")
    command: Optional[str] = Field(None, description="Optional command (e.g., 'finish')")

class GenerateSpeechRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    speaker: str = Field("default", description="Speaker voice to use")

# Initialize FastAPI app
app = FastAPI(title="MinoHealth Unified Speech Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components at startup
@app.on_event("startup")
async def startup_event():
    config = UnifiedSpeechConfig()
    logger = UnifiedLogger("unified_speech_service").logger
    model_manager = ModelManager(config, logger)

    await model_manager.initialize_models()

    app.state.config = config
    app.state.logger = logger
    app.state.model_manager = model_manager
    app.state.processor = AudioProcessor(config, model_manager, logger)

    # Start ngrok tunnel
    logger.info("Starting ngrok tunnel...")
    public_url = ngrok.connect(config.PORT, "http")
    app.state.ngrok_url = public_url.public_url
    app.state.ngrok_tunnel = public_url
    logger.info(f"ngrok tunnel established at: {app.state.ngrok_url}")

@app.on_event("shutdown")
async def shutdown_event():
    app.state.logger.info("Shutting down server and closing ngrok tunnel...")
    if hasattr(app.state, 'ngrok_tunnel'):
        ngrok.disconnect(app.state.ngrok_tunnel)
    app.state.logger.info("Server shutdown complete.")

@app.get("/")
async def root():
    """Root endpoint status check"""
    return {
        "status": "Server is running",
        "stt_url": f"{app.state.ngrok_url}/transcribe",
        "tts_url": f"{app.state.ngrok_url}/generate_speech"
    }

@app.post("/transcribe")
async def transcribe(request: Request):
    """STT endpoint"""
    processor = app.state.processor
    client_id = f"{request.client.host}:{request.client.port}"

    # Register client activity
    processor.register_client(client_id)

    try:
        data = await request.json()

        if 'command' in data and data['command'] == 'finish':
            app.state.logger.info(f"Client {client_id}: Finish command received")
            final_audio = processor.get_temp_audio()
            if len(final_audio) > 0:
                result = app.state.model_manager.stt_model.transcribe(
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
                    processor.accumulated_transcriptions.append(transcription)
                    return {"transcription": transcription, "is_final": True, "type": "final"}

            processor.clear_context()
            return {"status": "ok"}

        if 'audio' in data:
            audio_data = base64.b64decode(data['audio'])
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            processor.add_audio(audio_array, client_id)
            processor.update_client_stats(client_id, len(audio_array))  # Added this line to track audio samples

            if processor.should_process():
                audio_to_process = processor.get_audio()
                result = app.state.model_manager.stt_model.transcribe(
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
                    processor.update_context(transcription)
                    processor.accumulated_transcriptions.append(transcription)
                    processor.temp_buffer = np.array([], dtype=np.float32)
                    return {"transcription": transcription, "is_final": False, "type": "not-final"}

            return {
                "status": "ok",
                "buffer_size": len(processor.buffer),
                "session_info": {
                    "request_count": processor.clients[client_id]["request_count"],
                    "total_seconds": processor.clients[client_id]["total_audio_samples"]/16000
                }
            }

    except Exception as e:
        app.state.logger.error(f"Client {client_id}: Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_speech")
async def generate_speech(request: GenerateSpeechRequest):
    """TTS endpoint - FIXED: Removed 'await' since generate_speech is not async"""
    return app.state.processor.generate_speech(request.text, request.speaker)

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    except KeyboardInterrupt:
        print("\nShutdown requested... closing server")
    except Exception as e:
        print(f"Server error: {str(e)}")
    finally:
        print("Server stopped.")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    except KeyboardInterrupt:
        print("\nShutdown requested... closing server")
    except Exception as e:
        print(f"Server error: {str(e)}")
    finally:
        print("Server stopped.")