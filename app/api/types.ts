// API Types for MinoHealth CRM

// Audio Processing Types
export interface AudioTranscriptionRequest {
  audioData: string; // Base64 encoded audio data
  language?: string; // Optional language code
  processComplete?: boolean; // Flag to indicate this is a complete recording (not a streaming chunk)
  sampleRate?: number; // Sample rate of the audio in Hz
  sessionId?: string; // Unique identifier for the transcription session
  maintainContext?: boolean; // Whether to maintain context between requests
  alternativeMethod?: boolean; // Flag to use alternative transcription method when default fails
}

export interface AudioTranscriptionResponse {
  text: string;
  transcription?: string; // Alias for text for backward compatibility
  confidence?: number;
  segments?: TranscriptionSegment[];
  error?: string;
  isComplete?: boolean; // Flag to indicate if this is the final transcription
  sessionId?: string; // Session ID returned from the backend
  alternativeUsed?: boolean; // Whether an alternative transcription method was used
}

export interface TranscriptionSegment {
  text: string;
  start: number; // Start time in seconds
  end: number; // End time in seconds
  speaker?: string; // Optional speaker identification
}

export interface TextToSpeechRequest {
  text: string;
  speaker?: string; // Voice/speaker to use
  speed?: number; // Speech rate
}

export interface TextToSpeechResponse {
  audioData: string; // Base64 encoded audio data
  duration?: number; // Duration in seconds
  error?: string;
  sampleRate?: number; // Sample rate of the audio in Hz
}

// WebSocket Message Types
export interface WebSocketMessage {
  type: 'transcription' | 'speech' | 'audio' | 'error' | 'status' | 'message' | 'doctors' | 'summary' | 'appointment';
  payload?: any;
  timestamp?: number;
  text?: string; // For backward compatibility
  audio?: string; // For backward compatibility
  sample_rate?: number; // For backward compatibility
  data?: any; // For backward compatibility
}

export interface AudioResponse {
  audio: string; // Base64 encoded audio data
  sample_rate: number; // Sample rate in Hz
}

export interface TranscriptionStatus {
  isRecording: boolean;
  duration: number;
  status: 'idle' | 'recording' | 'processing' | 'done' | 'error';
}
