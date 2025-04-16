import requests
import json
import base64
import time
import threading
import queue
# Make PyAudio optional since we're not playing audio locally
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("PyAudio not available. Audio will not be played locally.")
import numpy as np
import argparse
import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TTSClient")

class TTSClient:
    def __init__(self, server_url):
        if not server_url:
            raise ValueError("Server URL is required")
        
        # Audio playback components
        self.server_url = server_url
        self.audio_queue = queue.Queue()
        self.is_playing = False
        
        # Only initialize PyAudio if available and we need local playback
        # (which we don't in this application)
        self.p = None
        if PYAUDIO_AVAILABLE:
            # We're keeping this code commented out but available for reference
            # self.p = pyaudio.PyAudio()
            pass
            
        self.stream = None
        self.stream_lock = threading.Lock()

        # Text processing components
        self.text_queue = queue.Queue()
        self.current_phrase = ""
        self.buffer_lock = threading.Lock()
        
        # Control flags
        self.stop_event = threading.Event()
        self.audio_finished_event = threading.Event()
        
        # Tracking for last audio chunk
        self.last_chunk_time = 0
        self.last_chunk_size = 0
        self.last_chunk_lock = threading.Lock()
        
        # Start processing threads
        self.text_thread = threading.Thread(target=self._process_text_queue, daemon=True)
        self.text_thread.start()
        
        # Start audio stream
        self.start_audio_stream()
        
        # Initialize audio chunks list
        self.all_audio_chunks = []
        
        # Initialize abbreviations
        self.abbreviations = {
            'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'Ph.D.', 'M.D.', 'B.A.', 'M.A.',
            'B.S.', 'M.S.', 'B.Sc.', 'M.Sc.', 'LL.B.', 'LL.M.', 'J.D.', 'Esq.', 'Inc.', 'Ltd.',
            'Co.', 'Corp.', 'Ave.', 'St.', 'Rd.', 'Blvd.', 'Dr.', 'Apt.', 'Ste.', 'No.', 'vs.',
            'etc.', 'i.e.', 'e.g.', 'a.m.', 'p.m.', 'U.S.', 'U.K.', 'N.Y.', 'L.A.', 'D.C.'
        }
        
    
    def start_audio_stream(self):
        """Start the audio playback stream"""
        # Comment out the actual audio stream creation since we're not playing locally
        """
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
        """
        # Instead, just set the playing state and clear the event
        self.is_playing = True
        self.audio_finished_event.clear()
        
        # Start a thread for processing audio (but not playing)
        self.play_thread = threading.Thread(target=self._play_audio, daemon=True)
        self.play_thread.start()
    
    def _play_audio(self):
        """Process audio chunks from the queue (but don't actually play)"""
        while self.is_playing and not self.stop_event.is_set():
            try:
                # Get audio chunk from queue with timeout
                audio_chunk = self.audio_queue.get(timeout=2.0)
                
                # Comment out the actual audio playback
                """
                # Ensure stream is open before playing
                with self.stream_lock:
                    if not self.is_playing or not self.stream or not self.stream.is_active():
                        self.start_audio_stream()
                    
                    try:
                        # Update last chunk info before playing
                        with self.last_chunk_lock:
                            self.last_chunk_time = time.time()
                            self.last_chunk_size = len(audio_chunk)
                        
                        # Play the audio
                        self.stream.write(audio_chunk)
                    except Exception as e:
                        # Reopen stream on error
                        logger.error(f"Error playing audio: {e}")
                        self.start_audio_stream()
                """
                
                # Still update the last chunk info for tracking
                with self.last_chunk_lock:
                    self.last_chunk_time = time.time()
                    self.last_chunk_size = len(audio_chunk)
                
                self.audio_queue.task_done()
            except queue.Empty:
                # No chunks available, check if we're done
                if self.audio_queue.empty() and self.text_queue.empty():
                    # Check if enough time has passed since last chunk was processed
                    with self.last_chunk_lock:
                        if self.last_chunk_time > 0 and self.last_chunk_size > 0:
                            last_chunk_duration = self.last_chunk_size / (24000 * 2)  # bytes / (sample_rate * bytes_per_sample)
                            time_since_last_chunk = time.time() - self.last_chunk_time
                            
                            # Only set finished event if the last chunk has had time to process
                            if time_since_last_chunk > last_chunk_duration:
                                # Signal that we've finished processing all audio
                                self.audio_finished_event.set()
                            else:
                                logger.info(f"Not enough time passed for last chunk to process, waiting...")
                        else:
                            # No audio has been processed yet or empty audio
                            self.audio_finished_event.set()
                continue
            except Exception as e:
                logger.error(f"Unexpected error in _play_audio: {e}")
                pass
        
        # Signal that we've finished processing all audio
        self.audio_finished_event.set()
    
    def _is_phrase_complete(self, text):
        """Check if a phrase is complete based on punctuation."""
        text = text.strip()
        phrase_endings = ['.', '?']
        return any(text.endswith(end) for end in phrase_endings)
    
    def _process_text_queue(self):
        """Process text chunks from the queue, sending complete phrases to TTS."""
        while not self.stop_event.is_set():
            try:
                # Get text chunk with timeout
                text_chunk = self.text_queue.get(timeout=0.1)
                
                with self.buffer_lock:
                    self.current_phrase += text_chunk
                    
                    #Check for abbreviations
                    if text_chunk.strip() in self.abbreviations:
                        continue
                    
                    # Check if phrase is complete
                    elif self._is_phrase_complete(self.current_phrase):
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
        if not text:
            return
        
        try:
            # Prepare the request
            url = f"{self.server_url}/tts-stream"
            logger.info(f"Sending text to TTS: {text}")
            
            response = requests.post(
                url,
                json={"text": text},
                headers={"Content-Type": "application/json"},
                stream=True
            )
            
            # Check if the request was successful
            if response.status_code == 200:
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
                                
                                # Add to all audio chunks for later use
                                self.all_audio_chunks.append(audio_bytes)
                                
                                # Add to the audio queue for playback
                                self.audio_queue.put(audio_bytes)
                                
                                # Update last chunk info
                                with self.last_chunk_lock:
                                    self.last_chunk_time = time.time()
                                    self.last_chunk_size = len(audio_bytes)
                        except json.JSONDecodeError:
                            # Incomplete JSON, continue adding to buffer
                            continue
            else:
                logger.error(f"Error from TTS server: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error sending text to TTS: {e}")
    
    def reset(self):
        """Reset the client to a clean state after stopping"""
        logger.info("Resetting TTS client state")
        
        # Clear the stop event to allow processing to continue
        self.stop_event.clear()
        self.audio_finished_event.clear()
        
        # Reset playing state
        with self.stream_lock:
            self.is_playing = True
        
        # Clear audio chunks
        self.all_audio_chunks = []
        
        # Restart processing threads if they've exited
        if not self.text_thread or not self.text_thread.is_alive():
            self.text_thread = threading.Thread(target=self._process_text_queue, daemon=True)
            self.text_thread.start()
            logger.info("Restarted text processing thread")
        
        if not self.play_thread or not self.play_thread.is_alive():
            self.play_thread = threading.Thread(target=self._play_audio, daemon=True)
            self.play_thread.start()
            logger.info("Restarted audio processing thread")

    def stream_text(self, text_chunk):
        """Add a chunk of text to the processing queue."""
        # Check if client was stopped - reset it before processing new text
        if self.stop_event.is_set():
            logger.info("Client was stopped, resetting before processing new text")
            self.reset()
        
        logger.info(f"Received: {text_chunk}")
        words = text_chunk.strip().split()
        if len(words) > 1:
            for word in words:
                self.text_queue.put(word + " ")
        else:
            self.text_queue.put(text_chunk)
    
    def wait_for_completion(self, timeout=5):
        """Wait for all audio to finish playing.
        
        Args:
            timeout: Maximum time (in seconds) to wait for inactivity before considering the process complete.
                     This is NOT an absolute timeout from start to finish, but rather a timeout for inactivity.
        """
        print("I am starting")
        # Process any remaining text
        with self.buffer_lock:
            if self.current_phrase:
                phrase_to_speak = self.current_phrase.strip()
                self.current_phrase = ""
                self._send_to_tts(phrase_to_speak)
        
        # Start monitoring for inactivity immediately
        last_activity_time = time.time()
        start_time = time.time()
        
        # Initialize tracking variables
        text_queue_size_last = self.text_queue.qsize()
        audio_queue_size_last = self.audio_queue.qsize()
        last_queue_change_time = time.time()
        
        # Keep monitoring until we detect sufficient inactivity
        while True:
            # Check if queues have changed size (indicating activity)
            current_text_size = self.text_queue.qsize()
            current_audio_size = self.audio_queue.qsize()
            
            # Check if queue sizes have changed
            if current_text_size != text_queue_size_last or current_audio_size != audio_queue_size_last:
                last_queue_change_time = time.time()
                last_activity_time = time.time()  # Reset inactivity timer when queues change
                
                # Update last known sizes
                text_queue_size_last = current_text_size
                audio_queue_size_last = current_audio_size
            
            # Check audio chunk activity
            with self.last_chunk_lock:
                if self.last_chunk_time > 0:  # If we've processed any audio
                    # Calculate expected duration of last chunk
                    last_chunk_duration = self.last_chunk_size / (24000 * 2)  # bytes / (sample_rate * bytes_per_sample)
                    time_since_last_chunk = time.time() - self.last_chunk_time
                    
                    # Update last activity time if we've received a new chunk recently
                    if time_since_last_chunk < 0.5:  # Very recent activity
                        last_activity_time = time.time()
                    
                    # If the last chunk hasn't had time to finish processing, we're still active
                    if time_since_last_chunk < last_chunk_duration:
                        last_activity_time = max(last_activity_time, self.last_chunk_time + last_chunk_duration)
            
            # Check if we've been inactive for the timeout period
            current_inactivity_time = time.time() - last_activity_time
            
            # Check if both queues are empty AND we've been inactive for the timeout
            if (self.text_queue.empty() and self.audio_queue.empty() and 
                current_inactivity_time >= timeout):
                break
            
            # If queues are not empty but there's been no activity for an extended period (2x timeout)
            # This is a safety check in case processing stalls
            if current_inactivity_time >= (timeout * 2):
                break
            
            # Short sleep to avoid CPU spinning
            time.sleep(0.2)
            
            # Safety timeout - if total time exceeds 5 minutes, exit with warning
            if time.time() - start_time > 300:  # 5 minutes
                break
        
        # Now set the event if it's still not set
        if not self.audio_finished_event.is_set():
            self.audio_finished_event.set()
        
        # Add a small fixed delay to ensure any final audio has finished processing
        time.sleep(1.0)
        
        print("I am done")
        return True
    
    def stop(self):
        """Stop all processing and clean up resources."""
        
        # Now set stop event
        self.stop_event.set()
        
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # Comment out the audio stream stopping
        """
        # Stop audio
        with self.stream_lock:
            self.is_playing = False
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass
        
        # Only terminate PyAudio if it was initialized
        if self.p:
            self.p.terminate()
        """
        
        # Just set the playing state to false
        with self.stream_lock:
            self.is_playing = False
        
        print("I stopped it yipee")
