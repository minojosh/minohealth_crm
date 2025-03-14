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
from typing import Optional

class TTSClient:
    def __init__(self, server_url):
        if not server_url:
            raise ValueError("Server URL is required")
        
        # Audio playback components
        self.server_url = server_url
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.stream_lock = threading.Lock()

        # Text processing components
        self.text_queue = queue.Queue()
        self.current_phrase = ""
        self.buffer_lock = threading.Lock()
        
        # Control flags
        self.stop_event = threading.Event()
        self.audio_finished_event = threading.Event()
        
        # Start processing threads
        self.text_thread = threading.Thread(target=self._process_text_queue, daemon=True)
        self.text_thread.start()
        
        # Start audio stream
        self.start_audio_stream()
        
        # Initialize audio chunks list
        self.all_audio_chunks = []
        
    
    def start_audio_stream(self):
        """Start the audio playback stream"""
        with self.stream_lock:
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass
                
            self.stream = self.p.open(
                format=self.p.get_format_from_width(2),  # 16-bit audio
                channels=1,
                rate=24000,
                output=True
            )
            self.is_playing = True
            self.audio_finished_event.clear()
        
        # Start a thread for playing audio
        self.play_thread = threading.Thread(target=self._play_audio, daemon=True)
        self.play_thread.start()
    
    def _play_audio(self):
        """Play audio chunks from the queue"""
        while self.is_playing and not self.stop_event.is_set():
            try:
                # Get audio chunk from queue with timeout
                audio_chunk = self.audio_queue.get(timeout=0.5)
                
                # Ensure stream is open before playing
                with self.stream_lock:
                    if not self.is_playing or not self.stream or not self.stream.is_active():
                        self.start_audio_stream()
                    
                    try:
                        self.stream.write(audio_chunk)
                    except Exception as e:
                        # Reopen stream on error
                        self.start_audio_stream()
                
                self.audio_queue.task_done()
            except queue.Empty:
                # No chunks available, check if we're done
                if self.audio_queue.empty() and self.text_queue.empty():
                    # Signal that we've finished playing all audio
                    self.audio_finished_event.set()
                continue
            except Exception:
                pass
        
        # Signal that we've finished playing all audio
        self.audio_finished_event.set()
    
    def _is_phrase_complete(self, text):
        """Check if a phrase is complete based on punctuation."""
        text = text.strip()
        phrase_endings = ['.', '!', '?', ':', ';']
        return any(text.endswith(end) for end in phrase_endings)
    
    def _process_text_queue(self):
        """Process text chunks from the queue, sending complete phrases to TTS."""
        while not self.stop_event.is_set():
            try:
                # Get text chunk with timeout
                text_chunk = self.text_queue.get(timeout=0.1)
                
                with self.buffer_lock:
                    self.current_phrase += text_chunk
                    
                    # Check if phrase is complete
                    if self._is_phrase_complete(self.current_phrase):
                        phrase_to_speak = self.current_phrase.strip()
                        self.current_phrase = ""
                        
                        # Send phrase to TTS service
                        self._send_to_tts(phrase_to_speak)
                
                self.text_queue.task_done()
            except queue.Empty:
                # If no new text but we have accumulated text, process it
                with self.buffer_lock:
                    if self.current_phrase and len(self.current_phrase.split()) >= 3:
                        phrase_to_speak = self.current_phrase.strip()
                        self.current_phrase = ""
                        # Send phrase to TTS service
                        self._send_to_tts(phrase_to_speak)
            except Exception:
                pass
    
    def _send_to_tts(self, text):
        """Send text to TTS service and process the response."""
        try:
            # Prepare request data
            data = {"text": text}
            
            
            # Make request to server
            response = requests.post(
                f"{self.server_url}/tts-stream",
                json=data,
                stream=True
            )
            
            if response.status_code != 200:
                return
            
            # Process streaming response
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
                        
                        if 'chunk' in data:
                            # Decode base64 audio chunk
                            audio_bytes = base64.b64decode(data['chunk'])
                            
                            # Add to audio queue for playback
                            self.audio_queue.put(audio_bytes)
                            
                            self.all_audio_chunks.append(audio_bytes)
                            
                    except json.JSONDecodeError:
                        # Incomplete JSON, continue adding to buffer
                        continue
        except Exception:
            pass
        
    
    def stream_text(self, text_chunk):
        """Add a chunk of text to the processing queue."""
        self.text_queue.put(text_chunk)
    
    def wait_for_completion(self, timeout=30):
        """Wait for all audio to finish playing."""
        # Process any remaining text
        with self.buffer_lock:
            if self.current_phrase:
                phrase_to_speak = self.current_phrase.strip()
                self.current_phrase = ""
                self._send_to_tts(phrase_to_speak)
        
        # Wait for text queue to empty
        start_time = time.time()
        while not self.text_queue.empty() and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        # Wait for audio queue to empty
        while not self.audio_queue.empty() and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        # Wait for audio finished event
        self.audio_finished_event.wait(timeout=timeout)
        
        # Additional delay to ensure last chunk is fully played
        time.sleep(2)
    
    def stop(self):
        """Stop all processing and clean up resources."""
        
        # Now set stop event
        self.stop_event.set()
        
        # Stop audio
        with self.stream_lock:
            self.is_playing = False
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass
        
        self.p.terminate()

def main():
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