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

// Track all active AudioContext instances
let activeAudioContexts: AudioContext[] = [];

// Function to register an AudioContext for tracking and cleanup
function registerAudioContext(ctx: AudioContext): void {
  activeAudioContexts.push(ctx);
  
  // Cleanup when it's closed
  ctx.addEventListener('statechange', () => {
    if (ctx.state === 'closed') {
      // Remove from tracking array when closed
      activeAudioContexts = activeAudioContexts.filter(c => c !== ctx);
    }
  });
}

// Function to close all active AudioContext instances
export function cleanupAllAudioContexts(): void {
  console.log(`Cleaning up ${activeAudioContexts.length} active AudioContext instances`);
  
  // Copy the array since we'll be modifying it during iteration
  const contextsToClean = [...activeAudioContexts];
  
  for (const ctx of contextsToClean) {
    try {
      if (ctx && ctx.state !== 'closed') {
        ctx.close().catch(err => {
          console.warn("Error closing AudioContext during cleanup:", err);
        });
      }
    } catch (err) {
      console.warn("Error during AudioContext cleanup:", err);
    }
  }
  
  // Clear the tracking array (will be updated via event listeners)
  activeAudioContexts = [];
}

// Clean up audio cache and contexts
export function cleanupAudio(): void {
  // Clear cached audio data
  clearAudioData();
  
  // Clean up any lingering audio contexts
  cleanupAllAudioContexts();
  
  console.log("Audio system cleaned up");
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
  if (!base64Audio) {
    console.error("No audio data provided");
    return;
  }
  
  let audioContext: AudioContext | null = null;
  
  try {
    console.log(`Playing audio with sample rate: ${sampleRate}Hz, encoding: ${encoding}, data length: ${base64Audio.length}`);
    
    // Create new AudioContext each time to avoid blocked contexts
    audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
      sampleRate: sampleRate // Try to match the incoming sample rate if possible
    });
    
    // Register for tracking and cleanup
    registerAudioContext(audioContext);
    
    const actualSampleRate = audioContext.sampleRate;
    console.log(`AudioContext initialized with sample rate: ${actualSampleRate}Hz`);
    
    try {
      // Ensure audio context is running (needed for newer browsers)
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
        console.log("AudioContext resumed from suspended state");
      }
    } catch (err) {
      console.warn("Error resuming audio context:", err);
    }
    
    // Better base64 decoding
    try {
      console.log("Decoding base64 data...");
      const binaryString = atob(base64Audio);
      console.log(`Decoded binary data length: ${binaryString.length} bytes`);
      
      // Use a more robust method to convert binary string to Uint8Array
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      console.log(`Uint8Array created with ${bytes.length} bytes`);
    
    // Process based on encoding with resampling if needed
    let audioData: Float32Array;
      
    if (encoding === 'PCM_16') {
        console.log("Processing as PCM_16 (Int16) audio...");
        
        // Verify we have enough bytes and they're even (2 bytes per sample for Int16)
        if (bytes.length < 2) {
          throw new Error(`Invalid audio data: too few bytes (${bytes.length})`);
        }
        
        if (bytes.length % 2 !== 0) {
          console.warn(`Audio data length (${bytes.length}) is not even, trimming last byte`);
          bytes.slice(0, bytes.length - 1);
        }
        
        // We need to handle alignment issues and endianness
        const dataView = new DataView(bytes.buffer);
        const samplesCount = Math.floor(bytes.length / 2);
        const int16Samples = new Int16Array(samplesCount);
        console.log(`Creating ${samplesCount} Int16 samples`);
        
        // Convert bytes to Int16 samples with correct endianness (little-endian)
        let nonZeroSamples = 0;
        for (let i = 0; i < samplesCount; i++) {
          const byteOffset = i * 2;
          if (byteOffset + 1 >= bytes.length) break;
          
          // Get Int16 (16-bit signed integer) value at the specified byte offset
          // with little-endian byte order (true parameter)
          const sampleValue = dataView.getInt16(byteOffset, true);
          int16Samples[i] = sampleValue;
          
          if (sampleValue !== 0) nonZeroSamples++;
        }
        
        console.log(`Got ${nonZeroSamples} non-zero samples out of ${samplesCount} total samples`);
        
        // Convert Int16 to Float32 (normalize to [-1, 1] range)
        // Int16 range is -32768 to 32767, so divide by 32768 for normalization
        audioData = new Float32Array(int16Samples.length);
        for (let i = 0; i < int16Samples.length; i++) {
          audioData[i] = int16Samples[i] / 32768.0;
        }
        
        // Apply minimal audio processing only if needed (default to false)
        // For TTS audio, we skip processing completely to maintain the original sound quality
        const shouldProcessAudio = false;
        if (shouldProcessAudio) {
          console.log("Applying minimal audio processing");
          audioData = processAudioForPlayback(audioData);
        } else {
          console.log("Skipping audio processing to maintain original quality");
          // Only apply gentle clipping prevention if needed
          let clippedSamples = 0;
          for (let i = 0; i < audioData.length; i++) {
            if (audioData[i] > 1.0) {
              audioData[i] = 1.0;
              clippedSamples++;
            } else if (audioData[i] < -1.0) {
              audioData[i] = -1.0;
              clippedSamples++;
            }
          }
          if (clippedSamples > 0) {
            console.log(`Prevented clipping on ${clippedSamples} samples`);
          }
        }
        
        // Log sample statistics for debugging
        const stats = calculateAudioStats(audioData);
        console.log("Audio statistics:", stats);
    } else {
      // Assume PCM_FLOAT (Float32)
        console.log("Processing as PCM_FLOAT (Float32) audio...");
        if (bytes.length % 4 !== 0) {
          console.warn(`Float32 data length (${bytes.length}) is not a multiple of 4, truncating`);
        }
        
        const samplesCount = Math.floor(bytes.length / 4);
        audioData = new Float32Array(samplesCount);
        const dataView = new DataView(bytes.buffer);
        
        for (let i = 0; i < samplesCount; i++) {
          const byteOffset = i * 4;
          if (byteOffset + 3 >= bytes.length) break;
          audioData[i] = dataView.getFloat32(byteOffset, true); // true = little endian
        }
    }
    
    // Create properly sized buffer with resampling if needed
    const resampleRatio = actualSampleRate / sampleRate;
    const buffer = audioContext.createBuffer(
      1, // mono
      Math.ceil(audioData.length * resampleRatio),
      actualSampleRate
    );
    
      console.log(`AudioBuffer created: ${buffer.length} samples at ${buffer.sampleRate}Hz`);
      
      // Apply resampling as needed
    const bufferData = buffer.getChannelData(0);
    if (Math.abs(resampleRatio - 1.0) < 0.001) {
      // No resampling needed, just copy the data
        console.log("No resampling needed, direct copy");
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
    
    // Create source node and start playback
    console.log("Creating source node and starting playback...");
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    
    // Register the source for tracking
    registerAudioSource(source);
    
    // Add gain node for volume control
    const gainNode = audioContext.createGain();
    gainNode.gain.value = 1.0; // Full volume
    
    // Connect the nodes: source -> gain -> destination
    source.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    // Setup playback completion Promise
    return new Promise<void>((resolve) => {
        let isResolved = false;
        let audioContextClosed = false;
        
        const markAsResolved = () => {
          if (!isResolved) {
            isResolved = true;
            resolve();
            
            // Clean up - only close if not already closed and after a delay
            if (!audioContextClosed) {
              audioContextClosed = true;
              
              // Wait a bit longer before closing to ensure all processing is complete
              setTimeout(() => {
                try {
                  if (audioContext && audioContext.state !== 'closed') {
                    console.log("Closing AudioContext after playback");
                    audioContext.close().catch(err => {
                      console.warn("Non-critical error while closing AudioContext:", err);
                    });
                  }
                } catch (err) {
                  console.warn("Error checking AudioContext state:", err);
                }
              }, 1000); // Longer timeout for reliable cleanup
            }
          }
        };

        source.onended = () => {
          console.log("Audio playback completed naturally");
          markAsResolved();
        };
        
        // Start playback
      source.start(0);
        console.log("Audio playback started");
      
      // Safety timeout in case onended doesn't fire
      const duration = buffer.duration * 1000;
        const safetyTimeout = Math.max(duration + 2000, 5000); // At least 5 seconds or duration + 2s
        console.log(`Setting safety timeout for ${Math.ceil(safetyTimeout)}ms`);
        
        setTimeout(() => {
          console.log("Audio playback timeout reached");
          markAsResolved();
        }, safetyTimeout);
      });
    } catch (error) {
      console.error("Error processing audio data:", error);
      throw error;
    }
  } catch (error) {
    console.error("Error playing audio:", error);
    
    // Close the AudioContext in case of error
    if (audioContext && audioContext.state !== 'closed') {
      try {
        await audioContext.close();
      } catch (closeError) {
        console.warn("Error closing AudioContext:", closeError);
      }
    }
    
    throw error;
  }
}

// Improved version with better noise filtering and normalization
function calculateAudioStats(audioData: Float32Array) {
  if (audioData.length === 0) return { min: 0, max: 0, mean: 0, maxAbs: 0, rms: 0 };
  
  let min = audioData[0];
  let max = audioData[0];
  let sum = 0;
  let maxAbs = 0;
  let sumSquared = 0;
  
  for (let i = 0; i < audioData.length; i++) {
    const value = audioData[i];
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    maxAbs = Math.max(maxAbs, Math.abs(value));
    sumSquared += value * value;
  }
  
  // Calculate RMS (Root Mean Square) - a better measure of audio loudness
  const rms = Math.sqrt(sumSquared / audioData.length);
  
  return {
    min,
    max,
    mean: sum / audioData.length,
    maxAbs,
    rms
  };
}

// Add this new function to improve audio quality for playback
function processAudioForPlayback(audioData: Float32Array): Float32Array {
  console.log("Applying enhanced audio processing for natural playback");
  
  // Create a copy to avoid modifying the original data
  const processed = new Float32Array(audioData.length);
  processed.set(audioData);
  
  // Calculate statistics to help with processing
  const stats = calculateAudioStats(processed);
  console.log("Pre-processing stats:", stats);

  // 1. Enhanced noise gate with smoother transition
  const noiseGateThreshold = 0.003; // Reduced threshold for more natural sound
  const smoothingFactor = 0.1;
  let prevSample = 0;
  let countBelowNoise = 0;
  
  for (let i = 0; i < processed.length; i++) {
    if (Math.abs(processed[i]) < noiseGateThreshold) {
      // Smooth transition to zero instead of hard gate
      processed[i] = prevSample * (1 - smoothingFactor);
      countBelowNoise++;
    }
    prevSample = processed[i];
  }
  
  console.log(`Enhanced noise gate applied: ${countBelowNoise} samples (${(countBelowNoise/processed.length*100).toFixed(1)}%) processed`);
  
  // 2. Normalize to use more of the dynamic range, but leave headroom
  let peak = 0;
  for (let i = 0; i < processed.length; i++) {
    peak = Math.max(peak, Math.abs(processed[i]));
  }
  
  if (peak > 0) {
    const targetPeak = 0.9; // Leave 10% headroom
    const scaleFactor = targetPeak / peak;
    for (let i = 0; i < processed.length; i++) {
      processed[i] *= scaleFactor;
    }
  }
  
  // 3. Soft knee compression for more natural dynamics
  const threshold = 0.4;
  const ratio = 0.7;
  const knee = 0.1;
  let compressedCount = 0;
  
  for (let i = 0; i < processed.length; i++) {
    const abs = Math.abs(processed[i]);
    if (abs > threshold - knee) {
      const overThreshold = abs - (threshold - knee);
      const compressed = (threshold - knee) + (overThreshold * ratio);
      processed[i] = Math.sign(processed[i]) * compressed;
      compressedCount++;
    }
  }
  
  console.log(`Soft knee compression applied to ${compressedCount} samples`);
  
  // 4. Apply a very gentle low-pass filter to smooth any remaining artifacts
  const alpha = 0.05; // Increased smoothing factor for better results
  let filtered = processed[0];
  for (let i = 1; i < processed.length; i++) {
    filtered = filtered * (1 - alpha) + processed[i] * alpha;
    processed[i] = filtered;
  }
  
  // Log final stats
  const finalStats = calculateAudioStats(processed);
  console.log("Post-processing stats:", finalStats);
  
  return processed;
}

// Track active audio source nodes for quick stopping
let activeAudioSources: AudioBufferSourceNode[] = [];

// Function to register an audio source for tracking
function registerAudioSource(source: AudioBufferSourceNode): void {
  activeAudioSources.push(source);
  
  // Remove from tracking when it ends
  source.onended = () => {
    activeAudioSources = activeAudioSources.filter(s => s !== source);
  };
}

/**
 * Immediately stop all currently playing audio
 * Simple function to silence audio when navigating away
 */
export function stopAllAudio(): void {
  console.log(`Stopping ${activeAudioSources.length} active audio sources`);
  
  // Stop all active audio sources
  for (const source of activeAudioSources) {
    try {
      if (source) {
        source.onended = null; // Remove listener to avoid cleanup issues
        source.stop();
      }
    } catch (err) {
      // Ignore errors when stopping already stopped sources
    }
  }
  
  // Clear the tracking array
  activeAudioSources = [];
  
  console.log("All audio playback stopped");
}
