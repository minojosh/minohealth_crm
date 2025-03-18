import { AudioTranscriptionRequest, AudioTranscriptionResponse, TextToSpeechRequest, TextToSpeechResponse, WebSocketMessage, AudioResponse } from './types';
import { API_ENDPOINTS, API_CONFIG } from './api';

class AudioService {
  private ws: WebSocket | null = null;
  private isConnected = false;
  private messageHandlers: Map<string, (payload: any) => void> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private audioBuffer: Float32Array[] = []; // Buffer to store audio chunks
  private lastAudioData: string | null = null; // Store the last processed audio data
  private currentSessionId: string | null = null; // Track the current session ID
  
  // Authentication token for WebSocket connections
  private authToken = "audio_access_token_2023";

  constructor(private readonly wsUrl: string = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000') {}

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error(`Maximum reconnection attempts (${this.maxReconnectAttempts}) reached`);
      return;
    }

    try {
      // Include the authentication token in the WebSocket URL
      const wsUrlWithAuth = `${this.wsUrl}/ws/audio?token=${this.authToken}`;
      console.log(`Connecting to WebSocket at ${wsUrlWithAuth} (Attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);
      this.ws = new WebSocket(wsUrlWithAuth);

      this.ws.onopen = () => {
        this.isConnected = true;
        this.reconnectAttempts = 0; // Reset reconnect attempts on successful connection
        console.log('WebSocket connected');
      };

      this.ws.onclose = (event) => {
        this.isConnected = false;
        console.log(`WebSocket disconnected with code: ${event.code}, reason: ${event.reason}`);
        
        // Don't reconnect if the connection was closed normally or if max attempts reached
        if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000); // Exponential backoff with 30s cap
          console.log(`Attempting to reconnect in ${delay/1000}s...`);
          setTimeout(() => this.connect(), delay);
        }
      };

      this.ws.onmessage = (event) => {
        try {
          console.log('Raw WebSocket message received:', event.data);
          const message: WebSocketMessage = JSON.parse(event.data);
          console.log('Parsed WebSocket message:', message);
          
          const handler = this.messageHandlers.get(message.type);
          if (handler) {
            // Check both formats: modern format has payload object, older format might have data directly
            const payloadData = message.payload || message;
            console.log(`Forwarding ${message.type} message payload:`, payloadData);
            handler(payloadData);
          } else {
            console.warn(`No handler registered for message type: ${message.type}`);
          }
        } catch (error) {
          console.error('Error processing WebSocket message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close(1000, 'Closing connection normally');
      this.ws = null;
      this.isConnected = false;
      this.reconnectAttempts = 0;
    }
  }

  async startTranscription(stream: MediaStream): Promise<MediaRecorder> {
    if (!this.isConnected) {
      console.log('WebSocket not connected, attempting to connect...');
      this.connect();
      
      // Wait for connection or timeout after 3 seconds
      const connected = await new Promise<boolean>((resolve) => {
        const timeout = setTimeout(() => resolve(false), 3000);
        const checkInterval = setInterval(() => {
          if (this.isConnected) {
            clearTimeout(timeout);
            clearInterval(checkInterval);
            resolve(true);
          }
        }, 100);
      });
      
      if (!connected) {
        console.error('Could not establish WebSocket connection');
        // Instead of failing, we'll continue as we'll use HTTP fallback later
        console.warn('Will use HTTP transcription as fallback');
      }
    }

    // Clear any previous audio buffer
    this.audioBuffer = [];
    
    // Generate new session ID for this transcription
    this.currentSessionId = crypto.randomUUID();
    console.log(`Starting new transcription session: ${this.currentSessionId}`);

    // Use a fixed sample rate for speech recognition
    const targetSampleRate = 16000; // 16kHz is optimal for speech recognition
    
    // Create an AudioContext with the target sample rate
    const audioContext = new AudioContext({
      sampleRate: targetSampleRate,
    });
    
    console.log(`Recording audio with sample rate: ${audioContext.sampleRate}Hz`);
    
    // Create a MediaStreamSource from the input stream
    const source = audioContext.createMediaStreamSource(stream);
    
    // Create a ScriptProcessor to get raw audio data
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    
    // Connect the audio processing chain
    source.connect(processor);
    processor.connect(audioContext.destination);

    // Process audio data with consistent format
    processor.onaudioprocess = (e) => {
      const inputData = e.inputBuffer.getChannelData(0);
      
      // Create a copy of the data to avoid reference issues
      const dataCopy = new Float32Array(inputData.length);
      dataCopy.set(inputData);
      
      // Apply a simple noise gate to reduce background noise
      for (let i = 0; i < dataCopy.length; i++) {
        // Simple noise gate - values below threshold are set to zero
        if (Math.abs(dataCopy[i]) < 0.01) {
          dataCopy[i] = 0;
        }
      }
      
      this.audioBuffer.push(dataCopy);
    };

    // Create a MediaRecorder to handle start/stop
    const mediaRecorder = new MediaRecorder(stream);
    
    // Handle stop event - process all audio at once
    mediaRecorder.onstop = async () => {
      console.log('Recording stopped, processing audio...');
      
      try {
        // Concatenate all audio chunks
        const totalLength = this.audioBuffer.reduce((acc, chunk) => acc + chunk.length, 0);
        console.log(`Processing ${totalLength} audio samples at ${audioContext.sampleRate}Hz`);
        
        if (totalLength === 0) {
          console.warn('No audio data to process');
          return;
        }
        
        // Concatenate all chunks into a single Float32Array
        const concatenated = new Float32Array(totalLength);
        let offset = 0;
        
        for (const chunk of this.audioBuffer) {
          concatenated.set(chunk, offset);
          offset += chunk.length;
        }
        
        // Process audio for transcription with better audio quality
        const processedAudio = this.processAudioForDirectTranscription(concatenated);
        
        // Convert to base64
        const base64Audio = this.arrayBufferToBase64(processedAudio.buffer);
        
        // Store the last processed audio data
        this.lastAudioData = base64Audio;
        
        // We'll skip the WebSocket approach and use direct HTTP API
        // since we've seen better results with that in our application
        try {
          const API_URL = process.env.NEXT_PUBLIC_SPEECH_SERVICE_URL || 'http://localhost:8000';
          const response = await fetch(`${API_URL}/transcribe`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              audio: base64Audio,
              sessionId: this.currentSessionId,
              maintain_context: false // Set to false to avoid context_prompt error
            })
          });
          
          if (response.ok) {
            const result = await response.json();
            console.log('Transcription successful:', result);
            
            // Check for error message in result
            if (result.error) {
              console.error("Transcription error from server:", result.error);
              
              // Detect the context_prompt error specifically
              if (result.error.includes("context_prompt")) {
                console.warn("Detected context_prompt error, using fallback method");
                // Retry with WebSocket as fallback
                if (this.ws?.readyState === WebSocket.OPEN) {
                  this.ws.send(JSON.stringify({
                    type: 'transcription',
                    payload: {
                      audioData: base64Audio,
                      format: 'raw',
                      codec: 'pcm',
                      sampleRate: audioContext.sampleRate,
                      sessionId: this.currentSessionId,
                      maintainContext: false,
                      processComplete: true
                    },
                    timestamp: Date.now(),
                    message_id: `transcription_${Date.now()}`
                  }));
                  return;
                }
              }
              
              // Let the error handler deal with it
              const errorHandler = this.messageHandlers.get('error');
              if (errorHandler) {
                errorHandler(result.error);
              }
              return;
            }
            
            // Manually trigger the transcription handler with the result
            const handler = this.messageHandlers.get('transcription');
            if (handler) {
              handler({ 
                text: result.transcription || '', 
                confidence: 1.0,
                sessionId: result.session_id || this.currentSessionId 
              });
            }
          } else {
            console.error('Direct transcription failed:', response.status);
            
            // Fallback to WebSocket if it's available
            if (this.ws?.readyState === WebSocket.OPEN) {
              console.log('Falling back to WebSocket transcription');
              this.ws.send(JSON.stringify({
                type: 'transcription',
                payload: {
                  audioData: base64Audio,
                  format: 'raw',
                  codec: 'pcm',
                  sampleRate: audioContext.sampleRate,
                  sessionId: this.currentSessionId,
                  maintainContext: false, // Set to false to avoid context_prompt error
                  processComplete: true
                },
                timestamp: Date.now(),
                message_id: `transcription_${Date.now()}`
              }));
            } else {
              // If both methods failed, tell the error handler
              const errorHandler = this.messageHandlers.get('error');
              if (errorHandler) {
                errorHandler('Failed to transcribe audio via all available methods');
              }
            }
          }
        } catch (err) {
          console.error('Error in direct transcription:', err);
          
          // Attempt WebSocket as a last resort
          if (this.ws?.readyState === WebSocket.OPEN) {
            console.log('Attempting WebSocket transcription as fallback');
            this.ws.send(JSON.stringify({
              type: 'transcription',
              payload: {
                audioData: base64Audio,
                format: 'raw',
                codec: 'pcm',
                sampleRate: audioContext.sampleRate,
                sessionId: this.currentSessionId,
                maintainContext: false, // Set to false to avoid context_prompt error
                processComplete: true
              },
              timestamp: Date.now(),
              message_id: `transcription_${Date.now()}`
            }));
          } else {
            const errorHandler = this.messageHandlers.get('error');
            if (errorHandler) {
              errorHandler('Failed to transcribe audio: ' + (err instanceof Error ? err.message : String(err)));
            }
          }
        }
        
        // Clear buffer after sending
        this.audioBuffer = [];
      } catch (error) {
        console.error('Error processing audio data:', error);
      } finally {
        // Clean up audio context
        try {
          processor.disconnect();
          source.disconnect();
          audioContext.close();
        } catch (err) {
          console.warn('Error cleaning up audio context:', err);
        }
      }
    };
    
    // Start recording
    mediaRecorder.start();
    
    return mediaRecorder;
  }

  // Enhanced audio processing for transcription
  private processAudioForDirectTranscription(audioData: Float32Array): Int16Array {
    // Calculate statistics for logging
    let min = 0, max = 0, sum = 0;
    for (let i = 0; i < audioData.length; i++) {
      min = Math.min(min, audioData[i]);
      max = Math.max(max, audioData[i]);
      sum += audioData[i];
    }
    
    console.log("Pre-processing audio stats:", {
      min, max, mean: sum / audioData.length
    });
    
    // Apply noise gate (filter out very low amplitudes)
    for (let i = 0; i < audioData.length; i++) {
      if (Math.abs(audioData[i]) < 0.005) {
        audioData[i] = 0;
      }
    }
    
    // Find maximum absolute value for effective normalization
    let maxAbs = 0;
    for (let i = 0; i < audioData.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(audioData[i]));
    }
    
    // If audio is too quiet (likely a silent recording), apply minimum threshold
    if (maxAbs < 0.01) {
      console.warn("Audio is too quiet, applying minimum threshold");
      maxAbs = 0.01; // Prevent division by zero or extreme amplification
    }
    
    // Calculate average level to ensure it meets server threshold
    let sumAbs = 0;
    for (let i = 0; i < audioData.length; i++) {
      sumAbs += Math.abs(audioData[i]);
    }
    const avgLevel = sumAbs / audioData.length;
    console.log("Audio average level:", avgLevel);
    
    // Server requires level >= 0.001, ensure we exceed this
    let boostFactor = 1.0;
    if (avgLevel < 0.001) {
      console.warn("Audio level too low for server processing, applying boost");
      boostFactor = 0.002 / avgLevel; // Boost to double the minimum requirement
    }
    
    // Normalize and convert to Int16Array with proper amplification
    const int16Data = new Int16Array(audioData.length);
    
    // Apply dynamic range compression for better audio
    const threshold = 0.3;
    const ratio = 0.6;
    
    for (let i = 0; i < audioData.length; i++) {
      // First normalize to [-1, 1] range
      let normalized = audioData[i] / maxAbs;
      
      // Apply simple dynamic range compression
      if (Math.abs(normalized) > threshold) {
        const overThreshold = Math.abs(normalized) - threshold;
        const compressed = overThreshold * ratio;
        normalized = (normalized > 0) ? 
          threshold + compressed : 
          -threshold - compressed;
      }
      
      // Apply any necessary boost
      normalized *= boostFactor;
      
      // Apply stronger amplification - increased from 0.95 to 0.98
      const amplified = normalized * 0.98;
      
      // Convert to 16-bit PCM range (-32768 to 32767)
      int16Data[i] = Math.max(-32768, Math.min(32767, amplified * 32767));
    }
    
    // Calculate post-processing stats for debugging
    let postMin = int16Data[0], postMax = int16Data[0], postSum = 0;
    for (let i = 0; i < int16Data.length; i++) {
      postMin = Math.min(postMin, int16Data[i]);
      postMax = Math.max(postMax, int16Data[i]);
      postSum += int16Data[i];
    }
    
    console.log("Post-processing audio stats:", {
      min: postMin, max: postMax, mean: postSum / int16Data.length
    });
    
    return int16Data;
  }

  // Helper method that safely converts ArrayBuffer to base64 without stack overflow
  private arrayBufferToBase64(buffer: ArrayBufferLike): string {
    const bytes = new Uint8Array(buffer);
    const len = bytes.length;
    let base64 = '';
    
    // Process in chunks of 16KB to avoid stack overflow
    const chunkSize = 24 * 1024;
    
    for (let i = 0; i < len; i += chunkSize) {
      const chunk = bytes.subarray(i, Math.min(i + chunkSize, len));
      base64 += String.fromCharCode.apply(null, Array.from(chunk));
    }
    
    return btoa(base64);
  }

  async synthesizeSpeech(text: string, speaker?: string): Promise<TextToSpeechResponse> {
    const request: TextToSpeechRequest = {
      text,
      speaker: speaker || 'default'
    };

    try {
      const response = await fetch(API_ENDPOINTS.tts, {
        method: 'POST',
        headers: API_CONFIG.headers,
        body: JSON.stringify(request)
      });

      if (!response.ok) {
        throw new Error(`Failed to synthesize speech: ${response.status} ${response.statusText}`);
      }

      return response.json();
    } catch (error) {
      console.error('Text-to-speech API error:', error);
      throw error;
    }
  }

  onTranscription(handler: (response: AudioTranscriptionResponse) => void) {
    this.messageHandlers.set('transcription', (payload) => {
      console.log('Handling transcription response:', payload);
      
      // Normalize payload to expected format with text/transcription field
      let normalizedPayload: AudioTranscriptionResponse;
      
      if (typeof payload === 'string') {
        normalizedPayload = { text: payload, confidence: 1.0 };
      } else if (payload.text) {
        normalizedPayload = payload;
      } else if (payload.transcription) {
        normalizedPayload = { 
          text: payload.transcription,
          confidence: payload.confidence || 0.9,
          isComplete: payload.isComplete || false
        };
      } else {
        console.warn('Unexpected transcription payload format:', payload);
        normalizedPayload = { 
          text: "Error: Unexpected response format",
          confidence: 0.0
        };
      }
      
      handler(normalizedPayload);
    });
  }

  onAudio(handler: (response: AudioResponse) => void) {
    this.messageHandlers.set('audio', (payload) => {
      console.log('Handling audio response:', payload);
      
      // Normalize payload to expected format with audio/sample_rate fields
      let normalizedPayload: AudioResponse;
      
      if (payload.audio && payload.sample_rate) {
        normalizedPayload = {
          audio: payload.audio,
          sample_rate: payload.sample_rate
        };
      } else {
        console.warn('Unexpected audio payload format:', payload);
        normalizedPayload = { 
          audio: "",
          sample_rate: 24000
        };
      }
      
      handler(normalizedPayload);
    });
  }

  onError(handler: (error: string) => void) {
    // Make sure error handler accepts string type
    this.messageHandlers.set('error', (payload) => {
      const errorMessage = typeof payload === 'string' 
        ? payload
        : payload?.message || 'Unknown error';
      handler(errorMessage);
    });
  }

  onStatus(handler: (status: string) => void) {
    this.messageHandlers.set('status', (payload) => {
      const statusMessage = payload.message || payload.status || 'Status update received';
      handler(statusMessage);
    });
  }

  getLastAudioData(): string | null {
    return this.lastAudioData;
  }

  // Get the current session ID
  getCurrentSessionId(): string | null {
    return this.currentSessionId;
  }

  // Static instance for singleton pattern
  private static instance: AudioService | null = null;
}

export const audioService = new AudioService();

/**
 * API utilities for handling audio in conversations
 */

// Cache to store audio data from backend conversations
let audioCache: { [key: string]: string } = {};

/**
 * Store base64 audio data in the cache
 * 
 * @param sessionId - Unique identifier for the conversation session
 * @param audioData - Base64 encoded audio data
 */
export function storeAudioData(sessionId: string, audioData: string): void {
  audioCache[sessionId] = audioData;
}

/**
 * Retrieve base64 audio data from the cache
 * 
 * @param sessionId - Unique identifier for the conversation session
 * @returns The base64 encoded audio data, or null if not found
 */
export function getAudioData(sessionId: string): string | null {
  return audioCache[sessionId] || null;
}

/**
 * Clear audio data from the cache
 * 
 * @param sessionId - Unique identifier for the conversation session to clear
 */
export function clearAudioData(sessionId?: string): void {
  if (sessionId) {
    delete audioCache[sessionId];
  } else {
    audioCache = {};
  }
}

// Singleton AudioContext with resample handling
const getAudioContext = (() => {
  let ctx: AudioContext | null = null;
  return (desiredSampleRate: number) => {
    if (!ctx || ctx.state === 'closed') {
      ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    return ctx;
  };
})();

/**
 * Play audio from base64 data with optimized handling to prevent glitches
 * 
 * @param base64Audio - Base64 encoded audio data
 * @param sampleRate - Sample rate of the audio data
 * @param encoding - Encoding format ('PCM_FLOAT' or 'PCM_16')
 * @param onUtteranceCreated - Optional callback that receives the utterance object for control
 * @returns Promise that resolves when audio playback is complete
 */
export async function playAudioFromBase64(
  base64Audio: string,
  sampleRate = 24000,
  encoding: 'PCM_FLOAT' | 'PCM_16' = 'PCM_FLOAT',
  onUtteranceCreated?: (utterance: SpeechSynthesisUtterance) => void
): Promise<void> {
  if (!base64Audio) return;
  
  try {
    console.log(`Playing audio with sample rate: ${sampleRate}Hz, encoding: ${encoding}`);
    
    const audioContext = getAudioContext(sampleRate);
    const actualSampleRate = audioContext.sampleRate;
    
    // Decode base64 efficiently
    const bytes = Uint8Array.from(atob(base64Audio), c => c.charCodeAt(0));
    
    // Process based on encoding with resampling if needed
    let audioData: Float32Array;
    if (encoding === 'PCM_16') {
      const int16Data = new Int16Array(bytes.buffer);
      audioData = new Float32Array(int16Data.length);
      for (let i = 0; i < int16Data.length; i++) {
        audioData[i] = int16Data[i] / 32768.0;
      }
    } else {
      // Assume PCM_FLOAT (Float32)
      audioData = new Float32Array(bytes.buffer);
    }
    
    // Create properly sized buffer with resampling if needed
    const resampleRatio = actualSampleRate / sampleRate;
    const buffer = audioContext.createBuffer(
      1, // mono
      Math.ceil(audioData.length * resampleRatio),
      actualSampleRate
    );
    
    // Apply antialiasing if resampling
    const bufferData = buffer.getChannelData(0);
    if (Math.abs(resampleRatio - 1.0) < 0.001) {
      // No resampling needed, just copy the data
      bufferData.set(audioData);
    } else {
      console.log(`Resampling audio from ${sampleRate}Hz to ${actualSampleRate}Hz (ratio: ${resampleRatio})`);
      // Simple linear resampling
      for (let i = 0; i < bufferData.length; i++) {
        const srcIdx = i / resampleRatio;
        const srcIdxFloor = Math.floor(srcIdx);
        const srcIdxCeil = Math.min(srcIdxFloor + 1, audioData.length - 1);
        const t = srcIdx - srcIdxFloor;
        bufferData[i] = audioData[srcIdxFloor] * (1 - t) + audioData[srcIdxCeil] * t;
      }
    }
    
    // Play with clean start/stop
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    
    return new Promise<void>((resolve) => {
      source.onended = () => resolve();
      source.start(0);
      
      // Safety timeout in case onended doesn't fire
      const duration = buffer.duration * 1000;
      setTimeout(() => resolve(), duration + 500);
    });
  } catch (error) {
    console.error("Error playing audio:", error);
    throw error;
  }
}
