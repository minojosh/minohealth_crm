"use client";

import React, { useState, useRef } from 'react';
import { ArrowLeftIcon, MicrophoneIcon, StopIcon, PaperAirplaneIcon } from '@heroicons/react/24/solid';
import { Card, CardBody } from "@heroui/card";
import { Button } from "@heroui/button";
import { Textarea } from "@heroui/input";
import DiagnosisSummary from '@/components/differential-diagnosis/DiagnosisSummary';
import useDiagnosisWebSocket from './useDiagnosisWebSocket';

// Config values that can be adjusted
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const AUDIO_CHUNK_SIZE = 30000; // 30 seconds of audio

interface DiagnosisConversationProps {
  patientId: number;
  onBack: () => void;
}

const DiagnosisConversation = ({ patientId, onBack }: DiagnosisConversationProps) => {
  // Use our custom WebSocket hook
  const {
    messages,
    error,
    isProcessing,
    diagnosisSummary,
    isFetchingSummary,
    connectionStatus,
    setError,
    setMessages,
    sendMessage,
    sendInitialContext,
    endConversation,
    setIsProcessing,
    reconnect
  } = useDiagnosisWebSocket(patientId);

  // State for recording
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [audioChunks, setAudioChunks] = useState<Blob[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const recordingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Local state
  const [inputText, setInputText] = useState('');

  // Scroll to bottom when messages change
  React.useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleStartRecording = async () => {
    try {
      // First make sure any existing recording is stopped
      if (isRecording) {
        handleStopRecording();
      }
      
      setIsRecording(true);
      setError(null);

      // Request microphone access with optimized settings
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,
          channelCount: 1
        }
      });
      
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
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        try {
          // Set processing state
          setIsProcessing(true);
          
          // Add a placeholder message that will be updated with the transcription
          setMessages(prev => [...prev, {
            role: 'user',
            text: 'Processing audio...',
            type: 'audio_processing',
            isPartial: true
          }]);
          
          // Concatenate all audio chunks
          const totalLength = audioBuffer.reduce((acc, chunk) => acc + chunk.length, 0);
          
          if (totalLength === 0) {
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

          // Find maximum absolute value for normalization
          let maxAbs = 0;
          for (let i = 0; i < concatenated.length; i++) {
            maxAbs = Math.max(maxAbs, Math.abs(concatenated[i]));
          }
          
          if (maxAbs > 0) {  // Avoid division by zero
            for (let i = 0; i < concatenated.length; i++) {
              concatenated[i] = concatenated[i] / maxAbs;
            }
          } else {
            setError('No audio signal detected');
            setIsProcessing(false);
            return;
          }

          // Check audio level after normalization
          let sumAbs = 0;
          for (let i = 0; i < concatenated.length; i++) {
            sumAbs += Math.abs(concatenated[i]);
          }
          const level = sumAbs / concatenated.length;

          if (level < 0.001) {
            setError('Audio level too low, please speak louder');
            setIsProcessing(false);
            return;
          }
          
          // Convert to base64
          const uint8Array = new Uint8Array(concatenated.buffer);
          let base64Data = '';
          
          for (let i = 0; i < uint8Array.length; i++) {
            base64Data += String.fromCharCode(uint8Array[i]);
          }
          
          base64Data = btoa(base64Data);
          
          // Strip trailing slashes and construct URL
          const baseUrl = process.env.NEXT_PUBLIC_STT_SERVER_URL?.replace(/\/+$/, '');
          const transcribeUrl = `${baseUrl}/transcribe`;

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
              // Send transcription as a message
              sendMessage(result.transcription);
            } else {
              setError('No speech detected');
              // Remove the processing message
              setMessages(prev => prev.filter(msg => !(msg.type === 'audio_processing' && msg.isPartial)));
              setIsProcessing(false);
            }
          } catch (error) {
            console.error("STT service error:", error);
            setError('Failed to process your recording');
            setIsProcessing(false);
          } finally {
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
      
      mediaRecorder.start(1000);
      
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

  const handleStopRecording = () => {
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

  const toggleRecording = () => {
    if (isRecording) {
      handleStopRecording();
    } else {
      handleStartRecording();
    }
  };

  const handleSendMessage = () => {
    if (inputText.trim()) {
      sendMessage(inputText);
      setInputText('');
    }
  };

  // Handle initial symptoms input if no messages yet
  const handleInitialSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) return;
    
    // Add initial symptoms as user message
    setMessages(prevMessages => [...prevMessages, {
      role: 'user',
      text: inputText,
      type: 'message'
    }]);
    
    // Send as context
    sendInitialContext(inputText);
    setInputText('');
  };

  // Render either the conversation or the diagnosis summary
  if (diagnosisSummary) {
    return (
      <Card className="h-full overflow-hidden">
        <CardBody className="flex flex-col p-0">
          <div className="p-4 flex justify-between items-center border-b">
            <Button
              variant="light"
              isIconOnly
              onClick={() => onBack()}
              className="mr-2"
            >
              <ArrowLeftIcon className="h-5 w-5" />
            </Button>
            <h2 className="text-xl font-semibold flex-1">Diagnosis Summary</h2>
          </div>
          <div className="flex-1 overflow-auto p-4">
            <DiagnosisSummary diagnosisResult={diagnosisSummary} />
          </div>
          <div className="p-4 border-t">
            <Button 
              color="primary" 
              className="w-full"
              onClick={onBack}
            >
              Back to Patients
            </Button>
          </div>
        </CardBody>
      </Card>
    );
  }

  return (
    <Card className="h-full overflow-hidden">
      <CardBody className="flex flex-col p-0">
        <div className="p-4 flex justify-between items-center border-b">
          <Button
            variant="light"
            isIconOnly
            onClick={onBack}
            className="mr-2"
          >
            <ArrowLeftIcon className="h-5 w-5" />
          </Button>
          <h2 className="text-xl font-semibold flex-1">Diagnosis Conversation</h2>
          <div className="flex items-center gap-2">
            <div className={`h-2 w-2 rounded-full ${
              connectionStatus === 'connected' ? 'bg-success' : 
              connectionStatus === 'connecting' ? 'bg-warning' : 
              'bg-danger'
            }`} />
            <span className="text-xs text-default-500">
              {connectionStatus === 'connected' ? 'Connected' : 
               connectionStatus === 'connecting' ? 'Connecting...' : 
               connectionStatus === 'disconnected' ? 'Disconnected' : 
               'Connection failed'}
            </span>
            {(connectionStatus === 'failed' || connectionStatus === 'disconnected') && (
              <Button 
                size="sm" 
                color="primary" 
                variant="light"
                onClick={reconnect}
              >
                Reconnect
              </Button>
            )}
          </div>
          <Button
            color="primary"
            variant="ghost"
            onClick={endConversation}
            disabled={isProcessing || messages.length <= 1 || isFetchingSummary || connectionStatus !== 'connected'}
          >
            {isFetchingSummary ? "Generating Summary..." : "Complete Diagnosis"}
          </Button>
        </div>
        
        <div className="flex-1 overflow-auto p-4">
          {messages.length === 0 && connectionStatus === 'connecting' && (
            <div className="flex justify-center items-center h-full">
              <div className="text-center">
                <div className="text-lg font-medium mb-2">Connecting to diagnosis service...</div>
                <div className="text-sm text-default-500">Please wait while we establish a connection.</div>
              </div>
            </div>
          )}
          
          {messages.length === 0 && connectionStatus === 'failed' && (
            <div className="flex justify-center items-center h-full">
              <div className="text-center">
                <div className="text-lg font-medium mb-2 text-danger">Connection failed</div>
                <div className="text-sm text-default-500 mb-4">Unable to connect to the diagnosis service. Please try again.</div>
                <Button color="primary" onClick={reconnect}>
                  Reconnect
                </Button>
              </div>
            </div>
          )}
          
          <div className="space-y-4">
            {messages.map((message, index) => (
              <div 
                key={index}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div 
                  className={`max-w-[80%] p-3 rounded-lg ${
                    message.role === 'user' 
                      ? 'bg-primary text-white' 
                      : 'bg-default-100'
                  } ${message.isPartial ? 'opacity-80' : ''}`}
                >
                  {message.type === 'audio_processing' ? (
                    <div className="flex items-center">
                      <span className="mr-2">🎤</span>
                      <span>{message.text}</span>
                    </div>
                  ) : (
                    <p style={{ whiteSpace: 'pre-wrap' }}>{message.text}</p>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>
        
        {error && (
          <div className="p-3 m-3 bg-danger-100 text-danger border border-danger rounded-md flex justify-between items-center">
            <div>{error}</div>
            {(connectionStatus === 'failed' || connectionStatus === 'disconnected') && (
              <Button 
                size="sm" 
                color="primary" 
                variant="flat"
                onClick={reconnect}
              >
                Reconnect
              </Button>
            )}
          </div>
        )}
        
        <div className="p-4 border-t">
          <form onSubmit={e => { e.preventDefault(); handleSendMessage(); }} className="flex w-full gap-3">
            <Textarea
              placeholder="Type your message..."
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              disabled={isProcessing || isRecording || isFetchingSummary || connectionStatus !== 'connected'}
              className="w-full h-full"
              rows={2}
              size='lg'
            />
            <div className="flex flex-col gap-2">
              {!isRecording ? (
                <Button
                  type="button"
                  color="primary"
                  variant="flat"
                  isIconOnly
                  onClick={toggleRecording}
                  disabled={isProcessing || isFetchingSummary || connectionStatus !== 'connected'}
                >
                  <MicrophoneIcon className="h-5 w-5" />
                </Button>
              ) : (
                <Button
                  type="button"
                  color="danger"
                  variant="ghost"
                  isIconOnly
                  onClick={toggleRecording}
                  disabled={isFetchingSummary}
                >
                  <StopIcon className="h-5 w-5" />
                </Button>
              )}
              <Button
                type="submit"
                color="primary"
                endContent={<PaperAirplaneIcon className="h-4 w-4" />}
                disabled={!inputText.trim() || isProcessing || isRecording || isFetchingSummary || connectionStatus !== 'connected'}
                isIconOnly
              >
              </Button>
            </div>
          </form>
        </div>
      </CardBody>
    </Card>
  );
};

export default DiagnosisConversation; 