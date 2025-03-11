"use client";

import { useState, useEffect, useRef } from 'react';
import { Button } from "@heroui/button";
import { audioService } from '../../app/api/audio';
import { TranscriptionStatus } from '../../app/api/types';

interface VoiceRecorderProps {
  onTranscriptionUpdate?: (text: string) => void;
  onRecordingStatusChange?: (status: TranscriptionStatus) => void;
  showTranscription?: boolean;
  autoConnect?: boolean;
  sampleRate?: number;
}

export default function VoiceRecorder({
  onTranscriptionUpdate,
  onRecordingStatusChange,
  showTranscription = false, // Changed from true to false to avoid duplicate display
  autoConnect = true,
  sampleRate = 16000, // Changed from 24000 to 16000 to match STT expectations
}: VoiceRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [status, setStatus] = useState<TranscriptionStatus>({
    isRecording: false,
    duration: 0,
    status: 'idle',
  });
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioVisualizerRef = useRef<HTMLCanvasElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Connect to WebSocket on component mount if autoConnect is true
  useEffect(() => {
    if (autoConnect) {
      audioService.connect();
    }

    // Set up event handlers
    audioService.onTranscription((response) => {
      console.log('Received transcription response:', response);
      
      // Check for error messages
      if (response.error) {
        console.error('Transcription error:', response.error);
        setError(response.error);
        return;
      }
      
      // Handle both response formats (text or transcription)
      const transcriptionText = response.text || response.transcription || '';
      
      if (transcriptionText) {
        console.log('Setting transcription:', transcriptionText);
        
        // Update the transcription text
        setTranscription(transcriptionText);
        
        // Notify parent component if callback exists
        if (onTranscriptionUpdate) {
          onTranscriptionUpdate(transcriptionText);
        }
        
        // If this is the final transcription after processing, update the status
        if (response.isComplete) {
          setIsProcessing(false);
          setStatus({
            isRecording: false,
            duration: status.duration,
            status: 'done',
          });
          
          if (onRecordingStatusChange) {
            onRecordingStatusChange({
              isRecording: false,
              duration: status.duration,
              status: 'done',
            });
          }
        }
      } else {
        console.log('No transcription text found in response:', response);
      }
    });

    audioService.onError((errorMsg) => {
      setError(errorMsg);
      setIsProcessing(false);
      stopRecording();
    });

    audioService.onStatus((statusMsg) => {
      console.log('Status update:', statusMsg);
    });

    // Clean up on unmount
    return () => {
      stopVisualization();
      if (mediaRecorderRef.current && isRecording) {
        mediaRecorderRef.current.stop();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, [autoConnect, onTranscriptionUpdate, onRecordingStatusChange, status.duration]);

  const startRecording = async () => {
    try {
      setError(null);
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000 // Request 16kHz sample rate for optimal speech recognition
        } 
      });
      
      // Start visualization
      startVisualization(stream);
      
      // Start recording
      const recorder = await audioService.startTranscription(stream);
      mediaRecorderRef.current = recorder;
      
      // Update state
      setIsRecording(true);
      setStatus({
        isRecording: true,
        duration: 0,
        status: 'recording',
      });
      
      // Notify parent component
      if (onRecordingStatusChange) {
        onRecordingStatusChange({
          isRecording: true,
          duration: 0,
          status: 'recording',
        });
      }
      
      console.log('Recording started');
    } catch (error) {
      console.error('Error starting recording:', error);
      setError(error instanceof Error ? error.message : 'Failed to start recording');
      stopVisualization();
    }
  };

  const stopRecording = () => {
    // Stop the media recorder if it exists
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
      
      // Update state
      setIsRecording(false);
      setIsProcessing(true);
      setStatus({
        isRecording: false,
        duration: 0,
        status: 'processing',
      });
      
      // Notify parent component
      if (onRecordingStatusChange) {
        onRecordingStatusChange({
          isRecording: false,
          duration: 0,
          status: 'processing',
        });
      }
      
      console.log('Recording stopped, processing audio...');
    }
    
    // Stop visualization
    stopVisualization();
  };

  const startVisualization = (stream: MediaStream) => {
    if (!audioVisualizerRef.current) return;
    
    try {
      // Create audio context for visualization only
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioCtx;
      
      // Create analyzer
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256; // Smaller FFT size for better performance
      analyser.smoothingTimeConstant = 0.8; // Smoother transitions
      analyserRef.current = analyser;
      
      // Connect stream to analyzer
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);
      
      // Start drawing
      drawVisualization();
    } catch (error) {
      console.error('Error starting visualization:', error);
    }
  };

  const drawVisualization = () => {
    if (!audioVisualizerRef.current || !analyserRef.current || !audioContextRef.current) return;
    
    const canvas = audioVisualizerRef.current;
    const canvasCtx = canvas.getContext('2d');
    if (!canvasCtx) return;
    
    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      // Stop if we're no longer recording
      if (!isRecording) return;
      
      // Schedule next frame
      animationFrameRef.current = requestAnimationFrame(draw);
      
      // Get frequency data
      analyser.getByteFrequencyData(dataArray);
      
      // Clear canvas
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Set up drawing
      const barWidth = (canvas.width / bufferLength) * 2.5;
      let x = 0;
      
      // Draw bars with gradient
      const gradient = canvasCtx.createLinearGradient(0, 0, 0, canvas.height);
      gradient.addColorStop(0, '#4f46e5'); // Indigo
      gradient.addColorStop(1, '#818cf8'); // Light indigo
      
      // Draw each bar
      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height;
        
        canvasCtx.fillStyle = gradient;
        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        
        x += barWidth + 1;
      }
    };
    
    // Start animation
    draw();
  };

  const stopVisualization = () => {
    // Cancel animation frame
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(err => {
        console.warn('Error closing audio context:', err);
      });
      audioContextRef.current = null;
    }
    
    // Clear canvas
    if (audioVisualizerRef.current) {
      const canvasCtx = audioVisualizerRef.current.getContext('2d');
      if (canvasCtx) {
        canvasCtx.clearRect(0, 0, audioVisualizerRef.current.width, audioVisualizerRef.current.height);
      }
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-center h-40 border-2 border-dashed rounded-lg border-default-200 relative">
        {isRecording ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <canvas 
              ref={audioVisualizerRef} 
              className="w-full h-full"
              width={300}
              height={150}
            />
          </div>
        ) : isProcessing ? (
          <div className="text-center">
            <p className="text-default-500">Processing your audio...</p>
            <div className="mt-2">
              <div className="w-8 h-8 border-4 border-t-primary border-r-transparent border-b-primary border-l-transparent rounded-full animate-spin mx-auto"></div>
            </div>
          </div>
        ) : (
          <div className="text-center">
            <p className="text-default-500">Click Start Recording to begin</p>
            <p className="text-sm text-default-400">Audio will be processed when you stop</p>
          </div>
        )}
      </div>
      
      <div className="flex justify-center gap-2">
        {!isRecording ? (
          <Button 
            color="primary" 
            onClick={startRecording}
            isDisabled={isProcessing}
            startContent={
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 15C13.66 15 15 13.66 15 12V6C15 4.34 13.66 3 12 3C10.34 3 9 4.34 9 6V12C9 13.66 10.34 15 12 15Z" fill="currentColor"/>
                <path d="M17 12C17 14.76 14.76 17 12 17C9.24 17 7 14.76 7 12H5C5 15.53 7.61 18.43 11 18.92V21H13V18.92C16.39 18.43 19 15.53 19 12H17Z" fill="currentColor"/>
              </svg>
            }
          >
            Start Recording
          </Button>
        ) : (
          <Button 
            color="danger" 
            onClick={stopRecording}
            startContent={
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M6 6H18V18H6V6Z" fill="currentColor"/>
              </svg>
            }
          >
            Stop Recording
          </Button>
        )}
      </div>
      
      {error && (
        <div className="text-red-500 text-sm mt-2">{error}</div>
      )}
      
      {showTranscription && transcription && (
        <div className="mt-4">
          <h3 className="text-md font-medium mb-2">Transcription:</h3>
          <p className="text-default-700">{transcription}</p>
        </div>
      )}
    </div>
  );
}
