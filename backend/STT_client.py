import asyncio
import pyaudio
import numpy as np
import base64
import os
import logging
from pynput import keyboard
import aiohttp
import threading
from dotenv import load_dotenv
import concurrent.futures

# Configure logger at module level
logger = logging.getLogger(__name__)

class SpeechRecognitionClient:
    def __init__(self, 
                 server_url=None,
                 rate=16000, 
                 channels=1, 
                 chunk=1600, 
                 max_queue_size=150, 
                 buffer_size=100):
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Connection settings - prioritize provided URL, then env var, then default
        if server_url:
            self.server_url = server_url.rstrip('/')
        else:
            env_url = os.getenv('STT_URL', '')
            if env_url:
                self.server_url = env_url.rstrip('/')
            else:
                # Fallback URL only if nothing else is provided
                self.server_url = "http://localhost:8000"
                logger.warning(f"No Speech Service URL provided. Using fallback URL: {self.server_url}")
                
        logger.info(f"Speech Service URL configured as: {self.server_url}")
        self.transcribe_url = f"{self.server_url}/transcribe"

        # Audio settings
        self.chunk = chunk
        self.channels = channels
        self.rate = rate
        self.max_chunk_samples = rate * 15  # Increased to 15 seconds max chunk

        # Audio processing objects
        self.p = None
        self.stream = None
        self.session = None

        # Queues and buffers
        self.audio_queue = None
        self.audio_buffer = asyncio.Queue(maxsize=buffer_size)
        self.current_chunk = []
        self.max_queue_size = max_queue_size

        # State flags
        self.running = True
        self.finish_pressed = False
        self.shutdown_event = asyncio.Event()
        self.is_processing = False  # Flag to track if we're currently processing audio

        # Control objects
        self.keyboard_listener = None
        self.tasks = set()

        # Transcription state
        self.transcription_text = ""
        self.last_printed = ""
        self.current_transcription = ""  # Track current transcription for frontend

        # Silence detection settings - increased for better user experience
        self.silence_threshold = 0.004  # Slightly lower to be more sensitive
        self.silence_duration = 0
        self.silence_timeout = 6  # Increased from 5 to 6 seconds
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def initialize_audio(self):
        """Initialize audio device and select appropriate microphone"""
        self.p = pyaudio.PyAudio()
        windows_input_devices = []

        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0:
                windows_input_devices.append((i, dev))
                

        device_index = None
        for idx, dev in windows_input_devices:
            if 'microphone' in dev['name'].lower():
                device_index = idx
                break

        if device_index is None and windows_input_devices:
            device_index = windows_input_devices[0][0]

        if device_index is None:
            raise Exception("No suitable audio input device found")

        return device_index

    async def _process_audio(self):
        """Process incoming audio data and put it in the queue"""
        def audio_callback(in_data, frame_count, time_info, status):
            if status:
                logging.debug(f"Status: {status}")
            try:
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                self.current_chunk.extend(audio_data.tolist())

                # Calculate audio level
                level = np.abs(audio_data).mean()

                # Silence detection
                if level < self.silence_threshold:
                    self.silence_duration += len(audio_data) / self.rate
                else:
                    self.silence_duration = 0

                # Send chunk if silence detected or max size reached
                if (0.8 <= self.silence_duration <= self.silence_timeout) or len(self.current_chunk) >= self.max_chunk_samples:
                    chunk_array = np.array(self.current_chunk, dtype=np.float32)
                    
                    if np.abs(chunk_array).mean() > 0.001:
                        try:
                            self.audio_queue.put_nowait(chunk_array)
                            self.current_chunk = []
                            self.silence_duration = 0
                            logging.info(f"Recorded {len(chunk_array)/self.rate:.1f} seconds")
                            self.is_processing = True
                        except asyncio.QueueFull:
                            logging.info("Audio queue full, dropping audio chunk")
                    else:
                        self.current_chunk = []
                elif self.silence_duration > self.silence_timeout: # Second trigger to close the connection to STT
                    logging.info("Silence timeout reached, finishing recording")
                    if len(self.current_chunk) > 0:
                        # Process any remaining audio before finishing
                        chunk_array = np.array(self.current_chunk, dtype=np.float32)
                        if np.abs(chunk_array).mean() > 0.001:
                            try:
                                self.audio_queue.put_nowait(chunk_array)
                                logging.info(f"Recorded final {len(chunk_array)/self.rate:.1f} seconds")
                            except asyncio.QueueFull:
                                logging.info("Audio queue full, dropping final audio chunk")
                    self.current_chunk = []
                    self.finish_pressed = True
                
                    
            except Exception as e:
                logging.error(f"Error in audio callback: {str(e)}", exc_info=True)

            return (in_data, pyaudio.paContinue)

        try:
            device_index = self.initialize_audio()
            self.audio_queue = asyncio.Queue(maxsize=self.max_queue_size)

            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk,
                stream_callback=audio_callback
            )

            print("Audio stream created, begin speaking ...")
            self.stream.start_stream()

            while self.running and not self.shutdown_event.is_set():
                if not self.audio_queue.empty():
                    try:
                        audio_data = await self.audio_queue.get()
                        await self.audio_buffer.put(audio_data)
                    except asyncio.QueueFull:
                        logging.info("Audio buffer full, dropping audio")
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logging.error(f"Error in audio processing: {e}")

    async def _send_audio(self, audio_data):
        """Send audio data to server for transcription"""
        try:
            message = {
                'audio': base64.b64encode(audio_data.tobytes()).decode('utf-8')
            }

            async with self.session.post(self.transcribe_url, json=message) as response:
                if response.status == 200:
                    result = await response.json()
                    transcription = result.get('transcription', None)

                    if transcription:
                        print(transcription, end='', flush=True)
                        # Store the transcription
                        self.transcription_text += transcription
                        # Update current transcription
                        self.current_transcription = transcription

                    return result
                else:
                    logging.error(f"Server error: {response.status}")
                    return None

        except Exception as e:
            logging.error(f"Error sending audio: {str(e)}", exc_info=True)
            raise

    async def _audio_buffer_consumer(self):
        """Consume audio from buffer and send to server"""
        while self.running and not self.shutdown_event.is_set():
            try:
                audio_data = await asyncio.wait_for(self.audio_buffer.get(), timeout=1.0)
                logging.info(f"Sending chunk of {len(audio_data)/self.rate:.1f} seconds")
                result = await self._send_audio(audio_data)
                # Mark that we're done processing this batch
                if self.audio_buffer.empty():
                    self.is_processing = False
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Error in audio consumer: {e}")
                self.is_processing = False
                break

    def on_press(self, key):
        """Handle keyboard events"""
        try:
            if key.char == 'f':
                self.finish_pressed = True
                logging.info("Finish signal triggered")
        except AttributeError:
            pass

    async def send_finish_signal(self):
        """Send finish signal to server"""
        try:
            message = {'command': 'finish'}
            async with self.session.post(self.transcribe_url, json=message) as response:
                if response.status == 200:
                    result = await response.json()
                    transcription = result.get('transcription', None)
                    if transcription:
                        print(transcription)
                        # Store the final transcription
                        self.transcription_text += transcription
                    logging.info("Sent finish signal")
        except Exception as e:
            logging.error(f"Error sending finish signal: {e}")

    async def close(self):
        """Graceful cleanup allowing tasks to complete"""
        logging.info("Starting graceful shutdown...")
        
        # Set the finish flag first to stop accepting new audio
        self.running = False
        self.shutdown_event.set()
        
        # Now close the audio stream
        if self.stream:
            logging.info("Closing audio stream...")
            self.stream.stop_stream()
            self.stream.close()

        if self.p:
            self.p.terminate()
        
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        
        # Send any audio chunks remaining in the queue to the buffer first
        if self.current_chunk:
            chunk_array = np.array(self.current_chunk, dtype=np.float32)
         
            if np.abs(chunk_array).mean() > 0.001:
                try:
                    self.audio_queue.put_nowait(chunk_array)
                    self.current_chunk = []
                except asyncio.QueueFull:
                    logging.info("Audio queue full, dropping audio chunk")
                    
        while not self.audio_queue.empty():
            try:
                audio_data = await self.audio_queue.get()
                await self.audio_buffer.put(audio_data)
            except asyncio.QueueFull:
                logging.info("Audio buffer full, dropping audio")
                await asyncio.sleep(0.01)
                
        if self.audio_buffer:
            logging.info("Processing remaining audio in buffer...")
            try:
                while not self.audio_buffer.empty():
                    audio = await asyncio.wait_for(self.audio_buffer.get(), timeout=1.0)
                    logging.info(f"Sending chunk of {len(audio)/self.rate:.1f} seconds")
                    await self._send_audio(audio)
                    # Give time for the server to process
                    await asyncio.sleep(0.3)
            except Exception as e:
                logging.error(f"Error processing final audio: {e}")
        try:
            # Send the finish signal and wait for final response
            logging.info("Sending finish signal...")
            await self.send_finish_signal()
            # Wait a moment for final results
            await asyncio.sleep(1.0) 
        except Exception as e:
            logging.error(f"Error in finish signal: {e}")   
        
        # Cancel any pending tasks
        pending_tasks = [t for t in self.tasks if not t.done()]
        if pending_tasks:
            logging.info(f"Waiting for {len(pending_tasks)} tasks to complete...")
            try:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            except Exception as e:
                logging.error(f"Error during task completion: {e}")
        
        # Finally close the session
        if self.session:
            await self.session.close()
            
        logging.info("Shutdown complete")
        self.is_processing = False  # Ensure processing flag is reset

    async def _run_recognition(self):
        """Run the speech recognition process"""
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        self.keyboard_listener.start()

        retry_count = 0
        max_retries = 3

        while retry_count < max_retries and self.running:
            try:
                async with aiohttp.ClientSession() as self.session:
                    print(f"Connecting to server at {self.server_url}")
                    print("Press 'f' to finish current transcription")

                    # Create and run tasks
                    self.tasks.update({
                        asyncio.create_task(self._process_audio()),
                        asyncio.create_task(self._audio_buffer_consumer()),
                    })

                    while self.running and not self.finish_pressed:
                        await asyncio.sleep(0.1)

                    if self.finish_pressed:
                        self.finish_pressed = False
                        await self.close()
                        break

            except aiohttp.ClientError as e:
                logging.error(f"\nAiohttp connection error: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    logging.info(f"Retrying in 5 seconds... (Attempt {retry_count + 1}/{max_retries})")
                    await asyncio.sleep(5)

            except Exception as e:
                logging.error(f"\nGeneral connection error: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    logging.info(f"Retrying in 5 seconds... (Attempt {retry_count + 1}/{max_retries})")
                    await asyncio.sleep(5)

    def recognize_speech(self):
        """Start speech recognition process"""
        # Set up asyncio for Windows if necessary
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Reset transcription text
        self.transcription_text = ""
        self.current_transcription = ""
        
        # Start the recognition process
        try:
            logging.info("Starting speech recognition...")
            asyncio.run(self._run_recognition())
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, shutting down...")
        
        return self.transcription_text

    def get_transcription(self):
        """Get the current transcription text"""
        return self.transcription_text
    
    def get_current_transcription(self):
        """Get the most recent transcription segment"""
        return self.current_transcription
        
    def is_audio_processing(self):
        """Check if audio is currently being processed"""
        return self.is_processing
        
    def transcribe_audio(self, audio_bytes):
        """
        Transcribe audio data received via WebSocket
        
        Args:
            audio_bytes (bytes): Raw audio bytes to transcribe
            
        Returns:
            str: Transcription result
        """
        try:
            logging.info(f"Received {len(audio_bytes)} bytes of audio data for transcription")
            
            # Create a thread-safe way to get the result from the async function
            result = None
            error = None
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            
            def run_async_in_thread():
                nonlocal result, error
                try:
                    # Create a new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    # Run the async function in this loop
                    result = loop.run_until_complete(self._send_audio_for_transcription(audio_bytes))
                    loop.close()
                except Exception as e:
                    error = str(e)
                    logging.error(f"Error in transcription thread: {e}")
            
            # Run the async code in a separate thread
            future = executor.submit(run_async_in_thread)
            future.result()  # Wait for the thread to complete
            executor.shutdown()
            
            if error:
                logging.error(f"Transcription failed: {error}")
                return f"Error during transcription: {error}"
            
            if result:
                logging.info(f"Transcription successful: {result}")
                self.current_transcription = result  # Store current transcription
                return result
            else:
                return "No transcription result received"
                
        except Exception as e:
            logging.error(f"Transcription error: {str(e)}")
            return "Transcription error occurred"
    
    async def _send_audio_for_transcription(self, audio_bytes):
        """Helper method to send audio data to the transcription server"""
        try:
            # Check if the audio data is valid for conversion to float32
            if len(audio_bytes) % 4 != 0:
                # If not divisible by 4 (float32 size), try to interpret as int16
                try:
                    # Convert from int16 to float32
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    logging.info(f"Converted audio from int16 to float32 format, shape: {audio_np.shape}")
                except ValueError as e:
                    logging.error(f"Error converting audio: {e}")
                    return f"Processing error: {str(e)}"
            else:
                # Try to interpret as float32
                try:
                    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                    logging.info(f"Interpreted audio as float32 format, shape: {audio_np.shape}")
                except ValueError as e:
                    # If that fails, try int16 as a fallback
                    try:
                        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        logging.info(f"Fallback: converted audio from int16 to float32, shape: {audio_np.shape}")
                    except ValueError as e2:
                        logging.error(f"Error converting audio: {e2}")
                        return f"Processing error: {str(e2)}"
            
            # Check audio level and normalize if needed
            max_val = np.abs(audio_np).max()
            if max_val > 0:  # Avoid division by zero
                if max_val > 1.0:  # If values are outside [-1, 1]
                    audio_np = audio_np / max_val  # Normalize to [-1, 1]
            
            # Ensure we have the right number of samples for float32
            if len(audio_np) % 4 != 0:
                padding = 4 - (len(audio_np) % 4)
                audio_np = np.pad(audio_np, (0, padding), 'constant')
            
            message = {
                'audio': base64.b64encode(audio_np.tobytes()).decode('utf-8')
            }
            
            # Check if the server URL is properly set before making the request
            if not self.server_url or not self.transcribe_url.startswith(('http://', 'https://')):
                error_msg = f"Invalid server URL: '{self.server_url}'. Please check your .env file and STT_SERVER_URL setting."
                logger.error(error_msg)
                return error_msg
                
            logger.info(f"Sending request to: {self.transcribe_url}")
            timeout = aiohttp.ClientTimeout(total=45)  # Increase timeout to 45 seconds for larger audio files
            
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    audio_duration = len(audio_np) / self.rate if hasattr(self, 'rate') else len(audio_np) / 16000
                    logger.info(f"Sending {len(audio_np)} samples ({audio_duration:.2f}s, level: {np.abs(audio_np).mean():.6f}) to transcription server")
                    
                    # Set the flag to indicate we're processing audio
                    self.is_processing = True
                    
                    async with session.post(self.transcribe_url, json=message) as response:
                        if response.status == 200:
                            try:
                                result_json = await response.json()
                                transcription = result_json.get('transcription', '')
                                logger.info(f"Received transcription response: {transcription}")
                                
                                # Set the flag to indicate we're done processing
                                self.is_processing = False
                                
                                # Return the transcription if it exists and isn't empty
                                if transcription and transcription.strip():
                                    self.current_transcription = transcription  # Store the current transcription
                                    return transcription
                                else:
                                    # Server returned no transcription - likely no speech was detected
                                    logger.warning("Empty transcription received from server")
                                    return "No speech detected"
                                    
                            except Exception as e:
                                logger.error(f"Error parsing server response: {str(e)}")
                                # Try to get the raw text if JSON parsing failed
                                text = await response.text()
                                logger.info(f"Raw server response: {text}")
                                
                                # If we got text but couldn't parse JSON, use the text as transcription
                                if text and not text.startswith(('Error', '{', '[')):
                                    self.current_transcription = text  # Store the current transcription
                                    return text
                                self.is_processing = False
                                return "Error parsing response"
                        else:
                            error_text = await response.text()
                            logger.error(f"Server error: {response.status}, {error_text}")
                            self.is_processing = False
                            return f"Server error {response.status}: {error_text}"
            except aiohttp.ClientConnectorError as ce:
                error_msg = f"Cannot connect to server at {self.transcribe_url}. Please check if the server is running and accessible."
                logger.error(f"{error_msg} Details: {str(ce)}")
                self.is_processing = False
                return error_msg
            except aiohttp.InvalidURL as iue:
                error_msg = f"Invalid URL format: {self.transcribe_url}. Please check your STT_SERVER_URL environment variable."
                logger.error(f"{error_msg} Details: {str(iue)}")
                self.is_processing = False
                return error_msg
            except asyncio.TimeoutError:
                logger.error(f"Request timed out after {timeout.total} seconds")
                self.is_processing = False
                return "Request timed out - server may be overloaded"
                
        except aiohttp.ClientError as e:
            logger.error(f"Connection error: {str(e)}")
            self.is_processing = False
            return f"Connection error: {str(e)}"
        
        except Exception as e:
            import traceback
            logger.error(f"Error in _send_audio_for_transcription: {str(e)}")
            logger.error(traceback.format_exc())
            self.is_processing = False
            return f"Processing error: {str(e)}"