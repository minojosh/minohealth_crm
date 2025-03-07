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
}

export default function VoiceRecorder({
  onTranscriptionUpdate,
  onRecordingStatusChange,
  showTranscription = false, // Changed from true to false to avoid duplicate display
  autoConnect = true,
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
      
      // Handle both response formats (text or transcription)
      const transcriptionText = response.text || response.transcription;
      
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
      setTranscription(''); // Clear previous transcription
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Start audio visualization
      startVisualization(stream);
      
      // Start recording
      mediaRecorderRef.current = await audioService.startTranscription(stream);
      
      setIsRecording(true);
      setIsProcessing(false);
      setStatus({
        isRecording: true,
        duration: 0,
        status: 'recording',
      });
      
      if (onRecordingStatusChange) {
        onRecordingStatusChange({
          isRecording: true,
          duration: 0,
          status: 'recording',
        });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start recording');
      console.error('Recording error:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      
      // Stop all tracks in the stream
      if (mediaRecorderRef.current.stream) {
        mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      }
      
      // Stop visualization
      stopVisualization();
      
      // Show processing state
      setIsRecording(false);
      setIsProcessing(true);
      setStatus({
        isRecording: false,
        duration: status.duration,
        status: 'processing',
      });
      
      if (onRecordingStatusChange) {
        onRecordingStatusChange({
          isRecording: false,
          duration: status.duration,
          status: 'processing',
        });
      }
    }
  };

  const startVisualization = (stream: MediaStream) => {
    if (!audioVisualizerRef.current) return;
    
    // Initialize audio context and analyzer
    audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    analyserRef.current = audioContextRef.current.createAnalyser();
    analyserRef.current.fftSize = 256;
    
    // Connect the stream to the analyzer
    const source = audioContextRef.current.createMediaStreamSource(stream);
    source.connect(analyserRef.current);
    
    // Start drawing the visualization
    drawVisualization();
  };

  const drawVisualization = () => {
    if (!audioVisualizerRef.current || !analyserRef.current) return;
    
    const canvas = audioVisualizerRef.current;
    const canvasCtx = canvas.getContext('2d');
    if (!canvasCtx) return;
    
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      if (!canvasCtx || !analyserRef.current) return;
      
      animationFrameRef.current = requestAnimationFrame(draw);
      
      analyserRef.current.getByteFrequencyData(dataArray);
      
      canvasCtx.fillStyle = 'rgb(240, 240, 240)';
      canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
      
      const barWidth = (canvas.width / bufferLength) * 2.5;
      let x = 0;
      
      for (let i = 0; i < bufferLength; i++) {
        const barHeight = dataArray[i] / 2;
        
        canvasCtx.fillStyle = `rgb(${barHeight + 100}, 134, 244)`;
        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        
        x += barWidth + 1;
      }
    };
    
    draw();
  };

  const stopVisualization = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    // Clear the canvas
    if (audioVisualizerRef.current) {
      const canvasCtx = audioVisualizerRef.current.getContext('2d');
      if (canvasCtx) {
        canvasCtx.clearRect(
          0, 
          0, 
          audioVisualizerRef.current.width, 
          audioVisualizerRef.current.height
        );
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
