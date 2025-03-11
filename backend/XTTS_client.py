import requests
import json
import base64
import time
import threading
import queue
import pyaudio
import numpy as np
import argparse
import os
from dotenv import load_dotenv
from pathlib import Path

class TTSClient:
    def __init__(self, server_url):
        # server_url = os.getenv("XTTS_URL")
        if not server_url:
            raise ValueError("Server URL is required")
        
        self.server_url = server_url
        self.audio_queue = queue.Queue()
        self.is_playing = False

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None

    def start_audio_stream(self):
        """Start the audio playback stream"""
        self.stream = self.p.open(
            format=self.p.get_format_from_width(2),  # 16-bit audio
            channels=1,
            rate=24000,
            output=True
        )
        self.is_playing = True

        # Start a thread for playing audio
        threading.Thread(target=self.play_audio, daemon=True).start()

    def play_audio(self):
        """Play audio chunks from the queue"""
        while self.is_playing:
            try:
                # Get audio chunk from queue with timeout
                audio_chunk = self.audio_queue.get(timeout=0.5)
                self.stream.write(audio_chunk)
                self.audio_queue.task_done()
            except queue.Empty:
                # No chunks available, but keep checking
                continue

    def stop_audio_stream(self):
        """Stop the audio playback stream"""
        self.is_playing = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

    def text_to_speech(self, text, language=None):
        """Send text to TTS service and play audio as it streams"""
        try:
            # Start audio stream before making request
            self.start_audio_stream()

            # Create a list to store all audio chunks for WAV file
            all_audio_chunks = []

            # Prepare request data
            data = {"text": text}
            if language is not None:
                data["language"] = language

            # Make streaming request to server
            print("Sending request to TTS server...")
            response = requests.post(
                f"{self.server_url}/tts-stream",
                json=data,
                stream=True
            )

            if response.status_code != 200:
                print(f"Error: Server returned status code {response.status_code}")
                print(response.text)
                return

            # Process streaming response
            print("Receiving audio stream...")
            buffer = ""
            for line in response.iter_lines():
                if line:
                    # Decode bytes to string
                    line_str = line.decode('utf-8')
                    buffer += line_str

                    # Try to parse JSON from buffer
                    try:
                        data = json.loads(buffer)
                        buffer = ""

                        # Decode base64 audio chunk
                        audio_bytes = base64.b64decode(data['chunk'])

                        # Add to audio queue for playback
                        self.audio_queue.put(audio_bytes)

                        # Store chunk for WAV file
                        all_audio_chunks.append(audio_bytes)

                    except json.JSONDecodeError:
                        # Incomplete JSON, continue adding to buffer
                        continue

            # Wait for all audio to be played
            print("Waiting for audio playback to complete...")
            self.audio_queue.join()
            time.sleep(0.5)  # Small delay to ensure last chunk is played
                    # Write complete audio to WAV file
            if all_audio_chunks:
                import wave
                filename = f"tts_output_{int(time.time())}.wav"
                with wave.open(filename, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(24000)  # 24kHz
                    wav_file.writeframes(b''.join(all_audio_chunks))
                print(f"Audio saved to {filename}")
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            self.stop_audio_stream()
            print("Audio playback complete.")

    def cleanup(self):
        """Clean up resources"""
        self.stop_audio_stream()
        self.p.terminate()

def main():
    # env_path = os.path.join(os.path.dirname(__file__).parent.parent, '.env')
    # env_path = "C:/Users/Mecha Mino 5 Outlook/Documents/Mino Health AI labs/minoHealth CRM/minohealth_crm/.env"
    # load_dotenv(dotenv_path=env_path) 
    tts_url = "https://47ea-34-16-148-110.ngrok-free.app"
    print(tts_url)
    client = TTSClient(server_url = tts_url)
    try:
        print("TTS Client - Enter text to convert to speech (type 'quit' to exit)")

        text = "Hello welcome to Moremi AI scheduler. I am listening to your voice. Please tell me what you want to do."
        client.text_to_speech(text)

    finally:
        client.cleanup()

if __name__ == "__main__":
    main()