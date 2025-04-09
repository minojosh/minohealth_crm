import { useState, useRef, useCallback, useEffect } from 'react';
import { audioService } from '../services/audio/AudioService';

interface AudioCaptureOptions {
  sampleRate?: number;
  channelCount?: number;
  echoCancellation?: boolean;
  noiseSuppression?: boolean;
  autoGainControl?: boolean;
}

interface AudioCaptureResult {
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  isRecording: boolean;
  error: string | null;
  audioData: {
    buffer: Float32Array[];
    lastAudioBase64: string | null;
  };
  stream: MediaStream | null;
  audioContext: AudioContext | null;
  clearAudioData: () => void;
}

export const useAudioCapture = (options: AudioCaptureOptions = {}): AudioCaptureResult => {
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Ref for current stream
  const streamRef = useRef<MediaStream | null>(null);
  
  // Cleanup function
  const cleanupAudio = useCallback(() => {
    // Use the AudioService to clean up resources
    audioService.stopRecording();
    streamRef.current = null;
  }, []);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanupAudio();
    };
  }, [cleanupAudio]);
  
  const clearAudioData = useCallback(() => {
    // Keep method for backward compatibility
  }, []);
  
  const startRecording = useCallback(async () => {
    // Clean up previous recording session
    cleanupAudio();
    setError(null);
    
    try {
      // Create audio constraints based on options
      const audioConstraints: MediaTrackConstraints = {
        echoCancellation: options.echoCancellation ?? true,
        noiseSuppression: options.noiseSuppression ?? true,
        autoGainControl: options.autoGainControl ?? true,
        channelCount: options.channelCount ?? 1,
        sampleRate: options.sampleRate ?? 16000
      };
      
      // Start recording using the AudioService
      const { stream } = await audioService.startRecording();
      streamRef.current = stream;
      
      setIsRecording(true);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to start recording');
      cleanupAudio();
    }
  }, [options, cleanupAudio]);
  
  const stopRecording = useCallback(() => {
    // Stop recording using the AudioService
    audioService.stopRecording();
    setIsRecording(false);
  }, []);
  
  return {
    startRecording,
    stopRecording,
    isRecording,
    error,
    audioData: {
      buffer: audioService.getAudioBuffer(),
      lastAudioBase64: audioService.getLastAudioData()
    },
    stream: streamRef.current,
    audioContext: audioService.audioContext,
    clearAudioData
  };
}; 