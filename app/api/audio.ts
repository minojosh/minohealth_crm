import { AudioTranscriptionRequest, AudioTranscriptionResponse, TextToSpeechRequest, TextToSpeechResponse, WebSocketMessage, AudioResponse } from './types';

class AudioService {
  private ws: WebSocket | null = null;
  private isConnected = false;
  private messageHandlers: Map<string, (payload: any) => void> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private audioBuffer: Float32Array[] = []; // Buffer to store audio chunks
  
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
        throw new Error('WebSocket connection failed');
      }
    }

    // Clear any previous audio buffer
    this.audioBuffer = [];

    // Create an AudioContext and set up processing chain
    const audioContext = new AudioContext({
      sampleRate: 16000, // Match the expected sample rate of the STT server
    });
    
    // Create a MediaStreamSource from the input stream
    const source = audioContext.createMediaStreamSource(stream);
    
    // Create a ScriptProcessor to get raw audio data
    // Note: ScriptProcessor is deprecated but still widely supported
    // We'll use it for now as the replacement AudioWorklet isn't as widely supported
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    
    // Connect the audio processing chain
    source.connect(processor);
    processor.connect(audioContext.destination);

    // Process audio data
    processor.onaudioprocess = (e) => {
      const inputData = e.inputBuffer.getChannelData(0);
      this.audioBuffer.push(new Float32Array(inputData));
    };

    // Create a MediaRecorder to handle start/stop
    const mediaRecorder = new MediaRecorder(stream);
    
    // Handle stop event - process all audio at once
    mediaRecorder.onstop = async () => {
      console.log('Recording stopped, processing audio...');
      
      try {
        // Concatenate all audio chunks
        const totalLength = this.audioBuffer.reduce((acc, chunk) => acc + chunk.length, 0);
        console.log(`Sending ${totalLength} audio samples for processing`);
        
        if (totalLength === 0) {
          console.warn('No audio data to process');
          return;
        }
        
        const concatenated = new Float32Array(totalLength);
        let offset = 0;
        
        for (const chunk of this.audioBuffer) {
          concatenated.set(chunk, offset);
          offset += chunk.length;
        }
        
        // Convert to base64 in chunks to avoid stack overflow
        const base64Audio = this.arrayBufferToBase64(concatenated.buffer);
        
        // Send complete audio to server
        if (this.ws?.readyState === WebSocket.OPEN) {
          const request: AudioTranscriptionRequest = {
            audioData: base64Audio,
            sampleRate: audioContext.sampleRate,
            processComplete: true // Flag to indicate this is the complete recording
          };
          
          this.ws.send(JSON.stringify({
            type: 'transcription',
            payload: request,
            timestamp: Date.now()
          }));
        } else {
          console.error('WebSocket not open, cannot send audio for processing');
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

  // Helper method that safely converts ArrayBuffer to base64 without stack overflow
  private arrayBufferToBase64(buffer: ArrayBuffer): string {
    const bytes = new Uint8Array(buffer);
    const len = bytes.length;
    let base64 = '';
    
    // Process in chunks of 16KB to avoid stack overflow
    const chunkSize = 16 * 1024;
    
    for (let i = 0; i < len; i += chunkSize) {
      const chunk = bytes.subarray(i, Math.min(i + chunkSize, len));
      base64 += String.fromCharCode.apply(null, Array.from(chunk));
    }
    
    return btoa(base64);
  }

  async synthesizeSpeech(text: string, speaker?: string): Promise<TextToSpeechResponse> {
    const request: TextToSpeechRequest = {
      text,
      speaker
    };

    try {
      const response = await fetch('/api/tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
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
      const statusMessage = typeof payload === 'string'
        ? payload
        : payload?.message || 'Unknown status';
      handler(statusMessage);
    });
  }
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

/**
 * Play audio from base64 data
 * 
 * @param base64Audio - Base64 encoded audio data
 * @param sampleRate - Optional sample rate for the audio data
 * @param onUtteranceCreated - Optional callback that receives the utterance object for control
 * @returns Promise that resolves when audio playback is complete
 */
export async function playAudioFromBase64(
  base64Audio: string, 
  sampleRate?: number,
  onUtteranceCreated?: (utterance: SpeechSynthesisUtterance) => void
): Promise<void> {
  if (!base64Audio) return;
  
  try {
    // First try to use the Web Speech API for more control
    if (window.speechSynthesis && onUtteranceCreated) {
      const utterance = new SpeechSynthesisUtterance("Text to speech playback");
      utterance.volume = 1;
      utterance.rate = 1;
      utterance.pitch = 1;

      // Call the callback with the utterance so caller can control it
      onUtteranceCreated(utterance);

      // Start speaking
      window.speechSynthesis.speak(utterance);
    }
    
    // Also play the actual audio data using AudioContext
    // Convert base64 to ArrayBuffer
    const binaryString = window.atob(base64Audio);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    
    // Create audio context and play audio
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
    
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    
    return new Promise((resolve) => {
      source.onended = () => resolve();
      source.start(0);
    });
  } catch (error) {
    console.error("Error playing audio:", error);
    throw error;
  }
}
