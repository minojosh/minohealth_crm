!pip install -q pyngrok
!pip install nest_asyncio
!pip install aiohttp
!pip install fastapi uvicorn TTS pydub
# Run the server
import nest_asyncio
import uvicorn
import base64
import io
import logging
import os
import numpy as np
import torch
import re
from pyngrok import ngrok
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
from google.colab import userdata
# ngrok.set_auth_token
auth_token = userdata.get('auth_token')
ngrok.set_auth_token(auth_token)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()
# Setup logging
logging.basicConfig(level=logging.INFO)

# Specify a default speaker name
SPEAKER_NAME = "cmu_us_slt_arctic"

# Initialize FastAPI
app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

class TTSModel:
    def __init__(self):
        self.model = None
        self.speaker_embedding = None
        self.gpt_cond_latent = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        try:
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            logging.info(f"⏳ Downloading model to {self.device}...")
            ModelManager().download_model(model_name)
            model_path = os.path.join(
                get_user_data_dir("tts"), model_name.replace("/", "--")
            )
            config = XttsConfig()
            config.load_json(os.path.join(model_path, "config.json"))
            self.model = Xtts.init_from_config(config)
            self.model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
            self.model.to(self.device)

            available_speakers = list(self.model.speaker_manager.speakers.keys())
            logging.info(f"Available speakers: {available_speakers}")

            speaker_to_use = SPEAKER_NAME if SPEAKER_NAME in available_speakers else available_speakers[0]
            logging.info(f"Using speaker: {speaker_to_use}")

            self.speaker_embedding = self.model.speaker_manager.speakers[speaker_to_use]["speaker_embedding"]
            self.gpt_cond_latent = self.model.speaker_manager.speakers[speaker_to_use]["gpt_cond_latent"]

            logging.info("🔥 Model Loaded")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    async def predict(self, text, language):
        if self.model is None:
            raise RuntimeError("Model not loaded. Please call load() first.")

        if not text or not text.strip():
            raise ValueError("Empty text provided")

        text_chunks = split_text(text, max_tokens=250)
        logging.info(f"Processing {len(text_chunks)} text chunks")

        for i, text_chunk in enumerate(text_chunks):
            logging.info(f"Processing chunk {i+1}/{len(text_chunks)}: {text_chunk[:50]}...")
            try:
                with torch.inference_mode():
                    outputs = self.model.inference(
                        text_chunk,
                        language=language,
                        gpt_cond_latent=self.gpt_cond_latent,
                        speaker_embedding=self.speaker_embedding,
                        enable_text_splitting=False
                    )
                # Convert to 16-bit PCM WAV format
                wav = np.clip(outputs["wav"], -1, 1)
                wav = (wav * 32767).astype(np.int16)
                yield wav
            except Exception as e:
                logging.error(f"Error processing chunk {i+1}: {str(e)}")
                raise

def split_text(text, max_tokens=250):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())

        if current_length + sentence_length > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    final_chunks = []
    for chunk in chunks:
        if len(chunk.split()) > max_tokens:
            words = chunk.split()
            for i in range(0, len(words), max_tokens):
                final_chunks.append(' '.join(words[i:i+max_tokens]))
        else:
            final_chunks.append(chunk)

    return final_chunks

# Initialize model
model = TTSModel()

@app.on_event("startup")
async def startup_event():
    model.load()

@app.get("/")
def health_check():
    return {"status": "healthy"}

@app.post("/tts-stream")
async def text_to_speech_base64(request: TTSRequest):
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        async def generate_audio():
            try:
                async for chunk in model.predict(request.text, request.language):
                    base64_chunk = base64.b64encode(chunk.tobytes()).decode('utf-8')
                    yield f'{{"chunk": "{base64_chunk}"}}\n'
            except Exception as e:
                logging.error(f"Error during audio generation: {str(e)}")
                yield f'{{"error": "{str(e)}"}}\n'
                raise

        return StreamingResponse(
            generate_audio(),
            media_type="application/json-stream"
        )
    except Exception as e:
        logging.error(f"Error in text_to_speech_base64: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
# Code to run in Colab
if __name__ == "__main__":

    # Get the public URL using ngrok for external access
    try:
        # Start ngrok tunnel
        public_url = ngrok.connect(8000)
        print(f"Public URL: {public_url}")
    except ImportError:
        print("For public access, install pyngrok: !pip install pyngrok")

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
    