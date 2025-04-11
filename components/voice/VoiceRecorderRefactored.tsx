import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from "@heroui/button";
import { useAudioCapture } from '../../hooks/useAudioCapture';
import { useTranscription } from '../../hooks/useTranscription';
import { TranscriptionStatus } from '../../app/api/types';
import AudioVisualizer from './core/AudioVisualizer';
import PulsingMicrophone from './core/PulsingMicrophone';
import { audioService } from '../../services/audio/AudioService';

interface VoiceRecorderProps {
  onTranscriptionUpdate?: (text: string, sessionId?: string) => void;
  onRecordingStatusChange?: (status: TranscriptionStatus) => void;
  showAudioVisualizer?: boolean;
  useHighQualityProcessing?: boolean;
}

const VoiceRecorder: React.FC<VoiceRecorderProps> = ({
  onTranscriptionUpdate,
  onRecordingStatusChange,
  showAudioVisualizer = true,
  useHighQualityProcessing = false
}) => {
  // Recording duration tracking
  const [recordingDuration, setRecordingDuration] = useState(0);
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null);
  
  // Initialize hooks
  const {
    startRecording,
    stopRecording,
    isRecording,
    error: captureError,
    stream,
    audioContext
  } = useAudioCapture({
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
    channelCount: 1,
    sampleRate: 16000
  });
  
  const {
    startTranscription,
    cancelTranscription,
    retryTranscription,
    transcription,
    isProcessing,
    error: transcriptionError,
    status,
    sessionId
  } = useTranscription({
    maxRetries: 3,
    timeoutMs: 15000,
    useHighQualityProcessing
  });
  
  // Combine errors
  const error = captureError || transcriptionError;
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
      audioService.cleanup();
    };
  }, []);
  
  // Update parent component with transcription
  useEffect(() => {
    if (transcription !== null && onTranscriptionUpdate) {
      onTranscriptionUpdate(transcription, sessionId);
    }
  }, [transcription, sessionId, onTranscriptionUpdate]);
  
  // Update parent component with status changes
  useEffect(() => {
    const updatedStatus = {
      ...status,
      duration: recordingDuration
    };
    
    if (onRecordingStatusChange) {
      onRecordingStatusChange(updatedStatus);
    }
  }, [status, recordingDuration, onRecordingStatusChange]);
  
  // Start recording duration counter
  const startDurationCounter = useCallback(() => {
    setRecordingDuration(0);
    
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
    }
    
    recordingTimerRef.current = setInterval(() => {
      setRecordingDuration(prev => prev + 0.1);
    }, 100);
  }, []);
  
  // Stop recording duration counter
  const stopDurationCounter = useCallback(() => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
  }, []);
  
  // Handle recording state
  const handleToggleRecording = useCallback(async () => {
    if (isRecording) {
      // Stop recording and process audio
      stopRecording();
      stopDurationCounter();
      
      // Set status to processing
      const newStatus: TranscriptionStatus = {
        isRecording: false,
        duration: recordingDuration,
        status: 'processing',
      };
      
      if (onRecordingStatusChange) {
        onRecordingStatusChange(newStatus);
      }
      
      // Start transcription - our updated hook will get audio data from AudioService
      await startTranscription();
    } else {
      // Start new recording
      try {
        await startRecording();
        startDurationCounter();
      } catch (err) {
        console.error("Failed to start recording:", err);
      }
    }
  }, [
    isRecording, 
    stopRecording, 
    stopDurationCounter, 
    onRecordingStatusChange, 
    recordingDuration, 
    startTranscription, 
    startRecording, 
    startDurationCounter
  ]);
  
  // Handle cancel button click
  const handleCancel = useCallback(() => {
    // First cancel any ongoing transcription API calls
    cancelTranscription();
    
    // Also clean up audio recording if it's still active
    if (isRecording) {
      stopRecording();
      stopDurationCounter();
    }
    
    // Reset state
    setRecordingDuration(0);
    audioService.cleanup();
    
    if (onRecordingStatusChange) {
      onRecordingStatusChange({
        isRecording: false,
        duration: 0,
        status: 'idle',
      });
    }
    
    if (onTranscriptionUpdate) {
      onTranscriptionUpdate('');
    }
  }, [
    cancelTranscription, 
    isRecording, 
    stopRecording, 
    stopDurationCounter, 
    onRecordingStatusChange, 
    onTranscriptionUpdate
  ]);
  
  // Reset recorder state
  const resetRecorder = useCallback(() => {
    cancelTranscription();
    audioService.cleanup();
    stopDurationCounter();
    setRecordingDuration(0);
    
    if (onTranscriptionUpdate) {
      onTranscriptionUpdate('');
    }
    
    if (onRecordingStatusChange) {
      onRecordingStatusChange({
        isRecording: false,
        duration: 0,
        status: 'idle',
      });
    }
  }, [
    cancelTranscription, 
    stopDurationCounter, 
    onTranscriptionUpdate, 
    onRecordingStatusChange
  ]);
  
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '8px',
        width: '100%',
      }}
    >
      {error && (
        <p style={{ color: '#d32f2f', margin: 0, fontSize: '0.875rem' }}>
          {error}
        </p>
      )}

      <PulsingMicrophone 
        isRecording={isRecording}
        isProcessing={isProcessing}
        onClick={handleToggleRecording}
        size={80}
        primaryColor="#1976d2"
        pulseColor="#9c27b0"
      />
      
      {(isProcessing || isRecording) && (
        <Button 
          color="danger" 
          onClick={handleCancel}
          className="mt-2 border border-danger-200"
          size="sm"
          startContent={
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          }
        >
          Cancel
        </Button>
      )}

      <p style={{ margin: 0, color: '#666', fontSize: '0.875rem' }}>
        {isRecording
          ? `Recording... ${recordingDuration.toFixed(1)}s`
          : isProcessing
          ? 'Processing audio...'
          : status.status === 'done'
          ? 'Transcription completed'
          : 'Click to record'}
      </p>
      
      {showAudioVisualizer && (
        <div style={{ marginTop: '8px', width: '300px', height: '60px' }}>
          <AudioVisualizer 
            audioContext={audioContext}
            stream={stream}
            isRecording={isRecording}
          />
        </div>
      )}
      
      {transcription && (
        <div
          style={{
            marginTop: '16px',
            padding: '16px',
            width: '100%',
            backgroundColor: 'rgba(0, 0, 0, 0.05)',
            borderRadius: '4px',
            maxHeight: '200px',
            overflowY: 'auto',
          }}
        >
          <p style={{ margin: 0 }}>{transcription}</p>
        </div>
      )}
    </div>
  );
};

export default VoiceRecorder; 