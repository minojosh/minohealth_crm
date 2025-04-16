
import numpy as np
import logging
from dotenv import load_dotenv
import asyncio
import requests
import json
import base64
import threading
import queue
import time
import re
from typing import Optional, Generator, AsyncGenerator, List, Union, Iterable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TTSClient")

class TTSClient:
    def __init__(self, server_url):
        if not server_url:
            raise ValueError("Server URL is required")
        
        self.server_url = server_url
        
        # Text processing components
        self.current_phrase = ""
        self.buffer_lock = threading.Lock()
        
        # Control flags
        self.stop_event = threading.Event()
        
        # Abbreviations list for text processing logic
        self.abbreviations = {
            'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'Ph.D.', 'M.D.', 'B.A.', 'M.A.',
            'B.S.', 'M.S.', 'B.Sc.', 'M.Sc.', 'LL.B.', 'LL.M.', 'J.D.', 'Esq.', 'Inc.', 'Ltd.',
            'Co.', 'Corp.', 'Ave.', 'St.', 'Rd.', 'Blvd.', 'Dr.', 'Apt.', 'Ste.', 'No.', 'vs.',
            'etc.', 'i.e.', 'e.g.', 'a.m.', 'p.m.', 'U.S.', 'U.K.', 'N.Y.', 'L.A.', 'D.C.'
        }
    
    def _should_process_current_phrase(self):
        """Determine if the current accumulated phrase should be processed."""
        if not self.current_phrase:
            return False
            
        # Check if the phrase ends with a sentence-ending punctuation
        if self.current_phrase.rstrip().endswith(('.', '!', '?')):
            # Make sure it's not just ending with an abbreviation
            for abbr in self.abbreviations:
                if self.current_phrase.rstrip().endswith(abbr):
                    return False
            return True
            
        # Process if we have enough words and there's a natural break
        # words = self.current_phrase.split()
        # return len(words) >= 5 and any(word.endswith((',', ';')) for word in words)
        
 
    def format_time_for_tts(self,text):
        # Pattern to match times like 11:00 AM, 3:45 PM, etc.
        pattern = r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)'
        
        def replace_time(match):
            hour = int(match.group(1))
            minute = int(match.group(2))
            period = match.group(3).upper()
            
            # Format the time in a TTS-friendly way
            if minute == 0:
                return f"{hour} o'clock {period}"
            else:
                return f"{hour} {minute} {period}"
        
        # Check if the pattern exists in the text
        if re.search(pattern, text):
            # If pattern exists, replace all occurrences
            return re.sub(pattern, replace_time, text)
        else:
            # If no pattern is found, return the original text
            return text

    
    def _send_to_tts(self, text):
        """Send text to TTS service and yield audio chunks."""
        if not text or not text.strip():
            return
        
        text = text.strip()
        # Format time in text
        text = self.format_time_for_tts(text)
        
        try:
            # Prepare the request
            url = f"{self.server_url}/tts-stream"
            logger.info(f"Sending text to TTS: {text}")
            
            response = requests.post(
                url,
                json={"text": text, "language": "en"},
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
                                yield audio_bytes
                        except json.JSONDecodeError:
                            # Incomplete JSON, continue adding to buffer
                            continue
            else:
               logger.error(f"Error from TTS server: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error sending text to TTS: {e}")
    
    def stream_text(self, text_chunk: str) -> Generator[bytes, None, None]:
        """
        Process a chunk of text and yield audio chunks.
        Works with both single words/tokens from LLM streaming and full paragraphs.
        """
        if not text_chunk:
            return
        
        logger.info(f"Received: {text_chunk}")
        
        words = text_chunk.strip().split()
        if len(words) > 1:
            word_list = []
            for word in words:
                word_list.append(word + " ")     
            words = word_list      

        for word in words:
            with self.buffer_lock:
                # Add text to current phrase buffer
                self.current_phrase += word
                
                # For LLM streaming input (typically one word/token at a time)
                # Check if the chunk is just a single period or another sentence-ending punctuation
                is_end_mark = text_chunk.strip() in ['.', '!', '?']
                
                # Skip processing if this is just an abbreviation
                is_abbreviation = text_chunk.strip() in self.abbreviations
                if is_abbreviation:
                    return
                    
                # Check if we should process what we have so far
                if self._should_process_current_phrase() or is_end_mark:
                    phrase_to_speak = self.current_phrase.strip()
                    self.current_phrase = ""

                    # Send phrase to TTS and yield audio chunks
                    yield from self._send_to_tts(phrase_to_speak)
    
    def process_text_stream(self, text_stream: Iterable[str]) -> Generator[bytes, None, None]:
        """Process a stream of text chunks and yield audio chunks."""
        for text_chunk in text_stream:
            if self.stop_event.is_set():
                break
                
            yield from self.stream_text(text_chunk)
            
        # Process any remaining text in the buffer if the stream is exhausted
        with self.buffer_lock:
            if self.current_phrase and not self.stop_event.is_set():
                yield from self._send_to_tts(self.current_phrase)
                self.current_phrase = ""
    
    def finish(self) -> Generator[bytes, None, None]:
        """Process any remaining text and signal completion."""
        with self.buffer_lock:
            if self.current_phrase and not self.stop_event.is_set():
                phrase_to_speak = self.current_phrase.strip()
                self.current_phrase = ""
                yield from self._send_to_tts(phrase_to_speak)
    
    def reset(self):
        """Reset the client state."""
        logger.info("Resetting TTS client state")
        with self.buffer_lock:
            self.current_phrase = ""
        self.stop_event.clear()
    
    def stop(self):
        """Stop all processing."""
        self.stop_event.set()
        with self.buffer_lock:
            self.current_phrase = ""
        logger.info("TTS client stopped")

    # Async implementations
    async def _send_to_tts_async(self, text):
        """Async version of send_to_tts."""
        if not text or not text.strip():
            return
        
        text = text.strip()
        
        try:
            # Prepare the request
            url = f"{self.server_url}/tts-stream"
            logger.info(f"Sending text to TTS (async): {text}")
            
            # Use httpx for async requests
            import httpx
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    url,
                    json={"text": text, "language": "en"},
                    headers={"Content-Type": "application/json"}
                ) as response:
                    # Check if the request was successful
                    if response.status_code == 200:
                        # Process streaming response
                        buffer = ""
                        
                        async for line in response.aiter_lines():
                            if line:
                                buffer += line
                                
                                # Try to parse JSON from buffer
                                try:
                                    data = json.loads(buffer)
                                    buffer = ""
                                    
                                    if 'chunk' in data:
                                        # Decode base64 audio chunk
                                        audio_bytes = base64.b64decode(data['chunk'])
                                        yield audio_bytes
                                except json.JSONDecodeError:
                                    # Incomplete JSON, continue adding to buffer
                                    continue
                    else:
                        logger.error(f"Error from TTS server: {response.status_code} - {await response.text()}")
        except Exception as e:
            logger.error(f"Error sending text to TTS async: {e}")
    
    async def stream_text_async(self, text_chunk: str) -> AsyncGenerator[bytes, None]:
        """Async version of stream_text."""
        if not text_chunk or self.stop_event.is_set():
            return
        
        logger.info(f"Received (async): {text_chunk}")
        
        with self.buffer_lock:
            # Add text to current phrase buffer
            self.current_phrase += text_chunk
            
            # For LLM streaming input (typically one word/token at a time)
            # Check if the chunk is just a single period or another sentence-ending punctuation
            is_end_mark = text_chunk.strip() in ['.', '!', '?']
            
            # Skip processing if this is just an abbreviation
            is_abbreviation = text_chunk.strip() in self.abbreviations
            if is_abbreviation:
                return
                
            # Check if we should process what we have so far
            if self._should_process_current_phrase() or is_end_mark:
                phrase_to_speak = self.current_phrase.strip()
                self.current_phrase = ""
                
                # Send phrase to TTS and yield audio chunks
                async for chunk in self._send_to_tts_async(phrase_to_speak):
                    yield chunk
    
    async def process_text_stream_async(self, text_stream) -> AsyncGenerator[bytes, None]:
        """Async version of process_text_stream."""
        async for text_chunk in text_stream:
            if self.stop_event.is_set():
                break
                
            async for chunk in self.stream_text_async(text_chunk):
                yield chunk
                
        # Process any remaining text in the buffer if the stream is exhausted
        with self.buffer_lock:
            if self.current_phrase and not self.stop_event.is_set():
                async for chunk in self._send_to_tts_async(self.current_phrase):
                    yield chunk
                self.current_phrase = ""
    
    async def finish_async(self) -> AsyncGenerator[bytes, None]:
        """Async version of finish."""
        with self.buffer_lock:
            if self.current_phrase and not self.stop_event.is_set():
                phrase_to_speak = self.current_phrase.strip()
                self.current_phrase = ""
                async for chunk in self._send_to_tts_async(phrase_to_speak):
                    yield chunk