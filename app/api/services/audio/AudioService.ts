/**
 * Centralized audio processing service
 * Handles audio recording, processing, and quality improvements
 */

class AudioService {
  private static instance: AudioService;
  private audioContext: AudioContext | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private audioSource: MediaStreamAudioSourceNode | null = null;
  private audioProcessor: ScriptProcessorNode | null = null;
  private audioBuffer: Float32Array[] = [];
  private currentSessionId: string | null = null;
  private lastAudioData: string | null = null;
  private currentStream: MediaStream | null = null;

  /**
   * Get the singleton instance
   */
  public static getInstance(): AudioService {
    if (!AudioService.instance) {
      AudioService.instance = new AudioService();
    }
    return AudioService.instance;
  }

  /**
   * Create audio context with optimal settings
   */
  public createAudioContext(sampleRate: number = 16000): AudioContext {
    if (!this.audioContext) {
      this.audioContext = new AudioContext({ sampleRate });
    }
    return this.audioContext;
  }

  /**
   * Request microphone access with optimal settings
   */
  public async requestMicrophone(options: MediaTrackConstraints = {}): Promise<MediaStream> {
    const defaultOptions: MediaTrackConstraints = {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
      sampleRate: 16000,
      ...options
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: defaultOptions
      });
      
      this.currentStream = stream;
      return stream;
    } catch (error) {
      console.error('Error accessing microphone:', error);
      throw error;
    }
  }

  /**
   * Process audio buffer with quality improvements
   */
  public processAudioBuffer(audioBuffer: Float32Array[] = this.audioBuffer): Uint8Array | null {
    if (!audioBuffer || audioBuffer.length === 0) {
      console.warn('No audio buffer to process');
      return null;
    }

    try {
      // Calculate total length and create concatenated buffer
      const totalLength = audioBuffer.reduce((acc, chunk) => acc + chunk.length, 0);
      const concatenated = new Float32Array(totalLength);
      
      let offset = 0;
      for (const chunk of audioBuffer) {
        concatenated.set(chunk, offset);
        offset += chunk.length;
      }

      // Apply noise gate
      for (let i = 0; i < concatenated.length; i++) {
        if (Math.abs(concatenated[i]) < 0.005) {
          concatenated[i] = 0;
        }
      }

      // Find maximum absolute value for normalization
      let maxAbs = 0;
      for (let i = 0; i < concatenated.length; i++) {
        maxAbs = Math.max(maxAbs, Math.abs(concatenated[i]));
      }

      // Apply minimum threshold for very quiet audio
      if (maxAbs < 0.01) {
        maxAbs = 0.01;
      }

      // Apply dynamic range compression and normalization
      for (let i = 0; i < concatenated.length; i++) {
        // Normalize to [-1, 1] range
        let normalized = concatenated[i] / maxAbs;
        
        // Apply compression for values exceeding threshold
        const threshold = 0.3;
        const ratio = 0.6; 
        
        if (Math.abs(normalized) > threshold) {
          const overThreshold = Math.abs(normalized) - threshold;
          const compressed = overThreshold * ratio;
          normalized = (normalized > 0) ? 
            threshold + compressed : 
            -threshold - compressed;
        }
        
        // Final normalization
        concatenated[i] = normalized * 0.95;
      }

      // Convert to 16-bit PCM
      const int16Data = new Int16Array(concatenated.length);
      for (let i = 0; i < concatenated.length; i++) {
        int16Data[i] = Math.max(-32768, Math.min(32767, concatenated[i] * 32767));
      }

      // Return as Uint8Array for further processing
      return new Uint8Array(int16Data.buffer);
    } catch (error) {
      console.error('Error processing audio buffer:', error);
      return null;
    }
  }

  /**
   * Start recording audio with processing
   */
  public async startRecording(sessionId?: string): Promise<{ stream: MediaStream; sessionId: string }> {
    // Clean up previous recording
    this.stopRecording();
    this.audioBuffer = [];
    this.lastAudioData = null;

    // Generate session ID if not provided
    this.currentSessionId = sessionId || crypto.randomUUID();

    try {
      // Request microphone access
      const stream = await this.requestMicrophone();
      
      // Create audio context
      const audioContext = this.createAudioContext();
      
      // Create audio source and processor
      const source = audioContext.createMediaStreamSource(stream);
      this.audioSource = source;
      
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      this.audioProcessor = processor;
      
      // Connect audio processing pipeline
      source.connect(processor);
      processor.connect(audioContext.destination);
      
      // Process audio data in chunks
      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Create a copy of the data to avoid reference issues
        const dataCopy = new Float32Array(inputData.length);
        dataCopy.set(inputData);
        
        // Apply simple noise gate
        for (let i = 0; i < dataCopy.length; i++) {
          if (Math.abs(dataCopy[i]) < 0.005) {
            dataCopy[i] = 0;
          }
        }
        
        // Store the processed audio chunk
        this.audioBuffer.push(dataCopy);
      };
      
      // Create MediaRecorder for backup
      try {
        const mediaRecorder = new MediaRecorder(stream, {
          mimeType: 'audio/webm;codecs=opus',
          audioBitsPerSecond: 16000
        });
        
        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            // Convert blob to base64 on stop
            const reader = new FileReader();
            reader.readAsDataURL(event.data);
            reader.onloadend = () => {
              const base64data = reader.result as string;
              // Remove the data URL prefix
              this.lastAudioData = base64data.split(',')[1];
            };
          }
        };
        
        // Start recording with 1s chunks
        mediaRecorder.start(1000);
        this.mediaRecorder = mediaRecorder;
      } catch (e) {
        console.warn('MediaRecorder not supported, using only ScriptProcessor');
      }
      
      return { 
        stream,
        sessionId: this.currentSessionId 
      };
    } catch (error) {
      console.error('Error starting recording:', error);
      throw error;
    }
  }

  /**
   * Stop recording and clean up resources
   */
  public stopRecording(): void {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
      this.mediaRecorder = null;
    }
    
    if (this.audioProcessor) {
      this.audioProcessor.disconnect();
      this.audioProcessor = null;
    }
    
    if (this.audioSource) {
      this.audioSource.disconnect();
      this.audioSource = null;
    }
    
    if (this.currentStream) {
      this.currentStream.getTracks().forEach(track => track.stop());
      this.currentStream = null;
    }
  }

  /**
   * Clean up all resources
   */
  public cleanup(): void {
    this.stopRecording();
    
    if (this.audioContext) {
      this.audioContext.close().catch(e => console.warn('Error closing AudioContext:', e));
      this.audioContext = null;
    }
    
    this.audioBuffer = [];
    this.lastAudioData = null;
    this.currentSessionId = null;
  }

  /**
   * Get the current audio buffer
   */
  public getAudioBuffer(): Float32Array[] {
    return this.audioBuffer;
  }

  /**
   * Get the last audio data as base64
   */
  public getLastAudioData(): string | null {
    return this.lastAudioData;
  }

  /**
   * Get the current session ID
   */
  public getCurrentSessionId(): string | null {
    return this.currentSessionId;
  }

  /**
   * Convert audio buffer to base64
   */
  public convertToBase64(audioData: Uint8Array): string {
    let binary = '';
    const len = audioData.byteLength;
    
    // Process in chunks to avoid stack overflow
    const chunkSize = 0xffff;
    
    for (let i = 0; i < len; i += chunkSize) {
      const chunk = audioData.slice(i, Math.min(i + chunkSize, len));
      const binaryChunk = String.fromCharCode.apply(null, [...chunk]);
      binary += binaryChunk;
    }
    
    return btoa(binary);
  }
}

// Export singleton instance
export const audioService = AudioService.getInstance();

// Also export the class for custom instances if needed
export default AudioService; 