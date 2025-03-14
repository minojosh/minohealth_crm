"use client";

import { useState, useEffect, useRef } from "react";
import { Card, CardBody } from "@heroui/card";
import { Button } from "@heroui/button";
import { MicrophoneIcon, StopIcon } from '@heroicons/react/24/solid';

interface Message {
  role: 'user' | 'system' | 'agent';
  content: string;
  timestamp: Date;
}

interface SchedulerConversationProps {
  initialContext: string;
  patientId: string;
  onComplete: () => void;
}

export function SchedulerConversation({ initialContext, onComplete, patientId }: SchedulerConversationProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  
  // WebSocket and Audio refs
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const MAX_RETRIES = 3;

  // Message handling
  const addMessage = (role: 'user' | 'agent', content: string) => {
    setMessages(prev => [...prev, {
      role,
      content,
      timestamp: new Date()
    }]);
  };

  // Add end conversation handler
  const endConversation = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log("Ending conversation...");
      wsRef.current.send(JSON.stringify({
        type: 'end_conversation'
      }));
      setIsProcessing(true);
      setError(null);
    } else {
      setError('Connection lost. Please refresh the page.');
    }
  };

  // WebSocket setup
  useEffect(() => {
    let isComponentMounted = true;

    const connectWebSocket = () => {
      if (!isComponentMounted || reconnectAttemptsRef.current >= MAX_RETRIES) {
        setError('Connection failed. Please refresh the page.');
        return;
      }

      const wsUrl = `${process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'}/ws/schedule/${patientId}`;
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        if (isComponentMounted) {
          setWsConnected(true);
          reconnectAttemptsRef.current = 0;
          setError(null);
          wsRef.current?.send(JSON.stringify({
            type: 'context',
            context: initialContext
          }));
        }
      };

      wsRef.current.onmessage = async (event) => {
        const data = JSON.parse(event.data);
        console.log('Received WebSocket message:', data);

        switch (data.type) {
          case 'message':
            // Add all messages (including system messages) to the chat
            addMessage(data.role || 'agent', data.text);
            if (data.role !== 'system') {
              setIsProcessing(false);
            }
            break;
          case 'summary':
            // Handle appointment summary
            if (data.text === "null") {
              addMessage('agent', "No appointment was scheduled.");
            } else {
              try {
                const summary = JSON.parse(data.text);
                addMessage('agent', `Appointment scheduled with ${summary.doctorName} on ${summary.appointmentDateTime}`);
              } catch (e) {
                console.error('Error parsing summary:', e);
              }
            }
            onComplete(); // Return to scheduler
            break;
          case 'error':
            setError(data.message);
            setIsProcessing(false);
            break;
          case 'status':
            setError(data.message);
            setIsProcessing(false);
            break;
          default:
            console.log('Unhandled message type:', data.type);
            break;
        }
      };

      wsRef.current.onclose = () => {
        if (isComponentMounted && wsConnected) {
          setWsConnected(false);
          reconnectAttemptsRef.current += 1;
          setTimeout(connectWebSocket, 2000);
        }
      };

      wsRef.current.onerror = () => {
        if (isComponentMounted) {
          setError('Connection error occurred.');
          setWsConnected(false);
        }
      };
    };

    connectWebSocket();
    return () => {
      isComponentMounted = false;
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [initialContext, patientId]);

  // Recording handlers
  const startRecording = async () => {
    console.log("Start recording called");
    
    // First make sure any existing recording is stopped
    if (isRecording) {
      console.log("Already recording, stopping first");
      stopRecording();
    }
    
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.error("WebSocket not connected");
      setError('Connection not established');
      return;
    }

    setIsRecording(true);
    setError(null);

    try {
      console.log("Requesting microphone access");
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,
          channelCount: 1
        }
      });
      
      console.log("Microphone access granted");
      streamRef.current = stream;
      
      // Create an AudioContext to process the audio
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      
      // Connect the audio nodes
      source.connect(processor);
      processor.connect(audioContext.destination);
      
      // Buffer to store raw audio data
      const audioBuffer: Float32Array[] = [];
      
      // Process audio data
      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        audioBuffer.push(new Float32Array(inputData));
      };
      
      // Create MediaRecorder for backup WebM recording
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 16000
      });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        console.log("Data available:", event.data.size);
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        console.log("MediaRecorder stopped, processing audio...");
        try {
          setIsProcessing(true);
          
          // Log initial audio buffer state
          console.log("Audio buffer chunks:", audioBuffer.length);
          audioBuffer.forEach((chunk, index) => {
            let chunkMin = chunk[0];
            let chunkMax = chunk[0];
            let sum = 0;
            for (let i = 0; i < chunk.length; i++) {
              chunkMin = Math.min(chunkMin, chunk[i]);
              chunkMax = Math.max(chunkMax, chunk[i]);
              sum += chunk[i];
            }
            console.log(`Chunk ${index} stats:`, {
              length: chunk.length,
              min: chunkMin,
              max: chunkMax,
              mean: sum / chunk.length
            });
          });
          
          // Concatenate all audio chunks
          const totalLength = audioBuffer.reduce((acc, chunk) => acc + chunk.length, 0);
          console.log(`Total audio length: ${totalLength} samples (${totalLength/16000} seconds)`);
          
          if (totalLength === 0) {
            console.warn('No audio data to process');
            setError('No audio data captured');
            setIsProcessing(false);
            return;
          }
          
          const concatenated = new Float32Array(totalLength);
          let offset = 0;
          
          for (const chunk of audioBuffer) {
            concatenated.set(chunk, offset);
            offset += chunk.length;
          }

          // Log pre-normalization stats
          let preMin = concatenated[0];
          let preMax = concatenated[0];
          let preSum = 0;
          let preNonZeroCount = 0;
          for (let i = 0; i < concatenated.length; i++) {
            preMin = Math.min(preMin, concatenated[i]);
            preMax = Math.max(preMax, concatenated[i]);
            preSum += concatenated[i];
            if (concatenated[i] !== 0) preNonZeroCount++;
          }
          const preNormStats = {
            length: concatenated.length,
            min: preMin,
            max: preMax,
            mean: preSum / concatenated.length,
            nonZeroCount: preNonZeroCount
          };
          console.log("Pre-normalization audio stats:", preNormStats);

          // Find maximum absolute value for normalization
          let maxAbs = 0;
          for (let i = 0; i < concatenated.length; i++) {
            maxAbs = Math.max(maxAbs, Math.abs(concatenated[i]));
          }
          console.log("Maximum absolute value:", maxAbs);
          
          if (maxAbs > 0) {  // Avoid division by zero
            for (let i = 0; i < concatenated.length; i++) {
              concatenated[i] = concatenated[i] / maxAbs;
            }
          } else {
            console.warn('Audio data contains all zeros');
            setError('No audio signal detected');
            setIsProcessing(false);
            return;
          }

          // Log post-normalization stats
          let postMin = concatenated[0];
          let postMax = concatenated[0];
          let postSum = 0;
          let postNonZeroCount = 0;
          for (let i = 0; i < concatenated.length; i++) {
            postMin = Math.min(postMin, concatenated[i]);
            postMax = Math.max(postMax, concatenated[i]);
            postSum += concatenated[i];
            if (concatenated[i] !== 0) postNonZeroCount++;
          }
          const postNormStats = {
            length: concatenated.length,
            min: postMin,
            max: postMax,
            mean: postSum / concatenated.length,
            nonZeroCount: postNonZeroCount
          };
          console.log("Post-normalization audio stats:", postNormStats);

          // Check audio level after normalization
          let sumAbs = 0;
          for (let i = 0; i < concatenated.length; i++) {
            sumAbs += Math.abs(concatenated[i]);
          }
          const level = sumAbs / concatenated.length;
          console.log(`Audio level after normalization: ${level}`);

          if (level < 0.001) {
            console.warn('Audio level too low after normalization:', level);
            setError('Audio level too low, please speak louder');
            setIsProcessing(false);
            return;
          }
          
          // Convert to base64
          const uint8Array = new Uint8Array(concatenated.buffer);
          let base64Data = '';
          
          console.log("Converting to base64...");
          console.log("Uint8Array length:", uint8Array.length);
          
          for (let i = 0; i < uint8Array.length; i++) {
            base64Data += String.fromCharCode(uint8Array[i]);
          }
          
          base64Data = btoa(base64Data);
          console.log("Base64 conversion complete. Length:", base64Data.length);
          
          // Strip trailing slashes and construct URL
          const baseUrl = process.env.NEXT_PUBLIC_STT_SERVER_URL?.replace(/\/+$/, '');
          const transcribeUrl = `${baseUrl}/transcribe`;
          console.log("Transcription URL:", transcribeUrl);

          try {
            // Send to STT service
            const response = await fetch(transcribeUrl, {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({ audio: base64Data })
            });

            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            if (result.transcription) {
              // Send transcription to scheduler WebSocket
              if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({
                  type: 'message',
                  role: 'user',
                  text: result.transcription
                }));
                
                // Add message locally for immediate feedback
                addMessage('user', result.transcription);
              } else {
                setError('Connection lost. Please refresh the page.');
              }
            } else {
              setError('No speech detected');
            }
          } catch (error) {
            console.error("STT service error:", error);
            setError('Failed to process your recording');
          } finally {
            setIsProcessing(false);
            
            // Clean up audio processing
            processor.disconnect();
            source.disconnect();
            audioContext.close();
          }
          
        } catch (error) {
          console.error("Error processing recording:", error);
          setError('Failed to process your recording');
          setIsProcessing(false);
        }
      };
      
      mediaRecorder.onerror = (event) => {
        console.error("MediaRecorder error:", event);
        setError('Recording error occurred');
        setIsRecording(false);
      };
      
      console.log("Starting MediaRecorder");
      mediaRecorder.start(1000);
      console.log("MediaRecorder started");
      
    } catch (error: any) {
      console.error("Error starting recording:", error);
      let errorMessage = 'Could not access microphone';
      
      if (error.name === 'NotAllowedError') {
        errorMessage = 'Microphone access denied. Please allow microphone access in your browser settings.';
      } else if (error.name === 'NotFoundError') {
        errorMessage = 'No microphone found. Please connect a microphone and try again.';
      }
      
      setError(errorMessage);
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    console.log("Stop recording called");
    
    // Set recording state to false immediately
    setIsRecording(false);
    
    try {
      // Stop the media recorder if it exists
      if (mediaRecorderRef.current) {
        if (mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.stop();
        }
      }
    } catch (error) {
      console.error("Error stopping media recorder:", error);
    }
    
    try {
      // Stop all tracks in the stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          track.stop();
        });
      }
    } catch (error) {
      console.error("Error stopping media tracks:", error);
    }
    
    // Clear references
    mediaRecorderRef.current = null;
    streamRef.current = null;
    audioChunksRef.current = [];
  };

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* Header with Back button and Status */}
      <div className="p-4 border-b bg-gray-900">
        <div className="flex items-center justify-between">
          {/* Back button */}
          <Button
            onClick={onComplete}
            className="bg-gray-600 hover:bg-gray-700 text-white"
          >
            ← Back to Scheduler
          </Button>

          {/* Status message - only shown when recording/processing */}
          {(isProcessing) && (
            <div className="flex items-center space-x-2 bg-blue-600/90 text-white px-4 py-2 rounded-lg">
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
              <span className="text-sm font-medium">
                Processing your message...
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Messages display */}
      <div className="flex-grow overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${
              message.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-4 py-2 ${
                message.role === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-800 text-gray-100 shadow-md'
              }`}
            >
              {message.content}
            </div>
          </div>
        ))}
      </div>

      {/* Recording controls at bottom */}
      <div className="p-4 bg-gray-900 border-t border-gray-700">
        <div className="flex justify-center gap-4">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={!isRecording && isProcessing}
            className={`rounded-lg px-6 py-3 flex items-center gap-2 ${
              isRecording 
                ? 'bg-red-600 hover:bg-red-700' 
                : 'bg-blue-600 hover:bg-blue-700'
            } text-white disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {isRecording ? (
              <>
                <StopIcon className="h-5 w-5" />
                Stop Recording
              </>
            ) : (
              <>
                <MicrophoneIcon className="h-5 w-5" />
                Start Recording
              </>
            )}
          </button>

          {/* Save Appointment button */}
          <button
            onClick={endConversation}
            disabled={!wsConnected || isProcessing || isRecording}
            className="rounded-lg px-6 py-3 flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
            </svg>
            Save Appointment
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 text-red-500">
          {error}
        </div>
      )}
    </div>
  );
}