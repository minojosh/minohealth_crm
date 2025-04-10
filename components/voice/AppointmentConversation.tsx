import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@heroui/button';
import { Card, CardBody } from '@heroui/card';
import { ReminderResponse } from '../../app/appointment-manager/types';
import { MicrophoneIcon, StopIcon, SpeakerWaveIcon, CheckCircleIcon, ClockIcon, BellIcon, ArrowPathIcon } from '@heroicons/react/24/solid';
import { appointmentManagerApi } from '../../app/appointment-manager/api';
import { playAudioFromBase64 } from '../../app/api/audio';
import { API_CONFIG } from '../../app/api/api';

interface Message {
  role: 'user' | 'system' | 'agent';
  content: string;
  timestamp: Date;
  audio?: string; // Base64 audio data for playback
  sampleRate?: number;
}

interface ConversationResult {
  status: string;
  [key: string]: any;
}

interface ConversationPanelProps {
  reminder: ReminderResponse;
  onComplete: (result: ConversationResult) => void;
  className?: string;
}

const arrayBufferToBase64 = (buffer: ArrayBuffer): string => {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
};

export function AppointmentConversation({ reminder, onComplete, className = "" }: ConversationPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [context, setContext] = useState<string>('');
  const [stage, setStage] = useState<'initial' | 'context' | 'conversation'>('initial');
  const [isContextSent, setIsContextSent] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isWaitingForInput, setIsWaitingForInput] = useState(false);
  const [availableSlots, setAvailableSlots] = useState<any[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  const [error, setError] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [audioData, setAudioData] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const latestAudioRef = useRef<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const lastPongTimeRef = useRef<number>(Date.now());
  
  // Add new state for conversation flow
  const [conversationState, setConversationState] = useState<'initial' | 'context' | 'confirmation'>('initial');
  
  // Add to existing state declarations
  const [appointmentDetails, setAppointmentDetails] = useState<any>(null);

  // Add text input functionality for testing
  const [textInput, setTextInput] = useState<string>('');

  const sendTextMessage = () => {
    if (!textInput.trim() || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }
    
    // Add user message to the UI
    addMessage('user', textInput);
    
    // Send message to the WebSocket
    wsRef.current.send(JSON.stringify({
      type: 'message',
      text: textInput
    }));
    
    // Clear input and set processing state
    setTextInput('');
    setIsProcessing(true);
  };

  // Message handling
  const addMessage = (role: 'user' | 'agent' | 'system', content: string) => {
    setMessages(prev => [...prev, {
      role,
      content,
      timestamp: new Date()
    }]);
  };

  // Connect to the conversation when a reminder is selected
  useEffect(() => {
    if (reminder) {
      console.log("Setting up conversation with reminder:", reminder);
      const id = `${reminder.message_type}_${reminder.details.patient_id || 'unknown'}_${Date.now()}`;
      setConversationId(id);
      
      // Force the initial state
      setConversationState('initial');
      console.log("Set conversation state to initial");
      
      // Add a small delay before connecting to ensure any previous connection is fully closed
      const timeoutId = setTimeout(() => {
        connectWebSocket(reminder.details.patient_id?.toString() || '1');
      }, 500);
      
      
      // Add welcome message
      const welcomeMessage = `Hello! I'm your appointment assistant. You have an appointment with Dr. ${reminder.details.doctor_name} on ${new Date(reminder.details.datetime).toLocaleString()}.`; 
     
      return () => {
        clearTimeout(timeoutId);
        if (wsRef.current) {
          wsRef.current.close();
        }
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
        }
      };
    }
    
    return () => {};
  }, [reminder]);

  // Keep track of the latest audio data in a ref
  useEffect(() => {
    latestAudioRef.current = audioData;
  }, [audioData]);

  // Auto-scroll to the latest message
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Reconnection handler
  const handleReconnection = (patientId: string) => {
    if (reconnectAttemptsRef.current < 5) {
      setTimeout(() => {
        reconnectAttemptsRef.current += 1;
        connectWebSocket(patientId);
      }, 2000 * Math.pow(2, reconnectAttemptsRef.current));
    }
  };


  const connectWebSocket = async (patientId: string) => {
    try {
      setConnectionStatus('connecting');
      
      // Create WebSocket connection to our appointment endpoint
      // Use a consistent ID format for the WebSocket connection
      const reminderId = reminder?.id || 'default';
      const wsUrl = `${process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'}/ws/${reminderId}`;
      
      console.log(`Connecting to WebSocket at: ${wsUrl}`);
      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        setConnectionStatus('connected');
        setError(null);
        
        // Send initial context to the WebSocket
        if (reminder) {
          console.log('Sending reminder context:', reminder);
          wsRef.current?.send(JSON.stringify({
            type: 'context',
            context: JSON.stringify(reminder)
          }));
        }
        
        // Set up client-side ping interval as a backup
        lastPongTimeRef.current = Date.now();
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
        }
        
        // Send a ping every 45 seconds (different from server's 30s to avoid collision)
        pingIntervalRef.current = setInterval(() => {
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            console.debug('Sending client ping');
            wsRef.current.send(JSON.stringify({ type: 'ping' }));
          }
        }, 45000);
      };
      
      wsRef.current.onmessage = async (event) => {
        const data = JSON.parse(event.data);
        console.log('Received WebSocket message:', data);
        
        switch (data.type) {
          case 'message':
            addMessage(data.role || 'agent', data.text);
            
            // If this is a goodbye message, set processing to true
            if (data.is_goodbye) {
              console.log('Received goodbye message, showing processing indicator');
              setIsProcessing(true);
            } else {
              setIsProcessing(false);
            }
            break;
            
          case 'ping':
            // Respond to server pings with pongs
            console.debug('Received ping from server, sending pong');
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({ type: 'pong' }));
            }
            // Update last pong time since we got activity from the server
            lastPongTimeRef.current = Date.now();
            break;
            
          case 'pong':
            // Update the last pong time
            console.debug('Received pong from server');
            lastPongTimeRef.current = Date.now();
            break;
            
          case 'appointment_result':
            // Handle the final appointment result from the backend
            if (data.success && data.appointment) {
              const appointment = data.appointment;
              setAppointmentDetails({
                ...appointment,
                status: appointment.status || 'confirmed'  // Use status from backend
              });
              // Let parent component know about the successful completion
              if (onComplete) {
                onComplete({
                  status: appointment.status || 'confirmed',
                  appointment: appointment,
                  success: true
                });
              }
            } else {
              const errorMsg = data.message || 'Failed to process appointment';
              setError(errorMsg);
              if (onComplete) {
                onComplete({
                  status: 'failed',
                  error: errorMsg,
                  success: false
                });
              }
            }
            setIsProcessing(false);
            break;
            
          case 'summary':
            if (onComplete) {
              onComplete(data.details || appointmentDetails || { status: 'completed' });
            }
            break;
            
          case 'error':
            setError(data.message || 'An error occurred');
            setIsProcessing(false);
            break;
            
          default:
            console.log('Unhandled message type:', data.type);
            break;
        }
      };
      
      wsRef.current.onclose = () => {
        if (connectionStatus === 'connected') {
          setConnectionStatus('disconnected');
          handleReconnection(patientId);
        }
        
        // Clear ping interval on close
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }
      };
      
      wsRef.current.onerror = () => {
        setConnectionStatus('error');
        setError('WebSocket connection error');
      };
    } catch (error) {
      console.error('Error connecting WebSocket:', error);
      setConnectionStatus('error');
      setError('Failed to establish connection');
    }
  };

  const handleStartRecording = async () => {
    try {
      // First make sure any existing recording is stopped
      if (isRecording) {
        handleStopRecording();
      }
      
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        setError('Connection not established');
        return;
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
          setIsProcessing(true);
          
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
              // Send transcription to WebSocket
              if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({
                  type: 'message',
                  text: result.transcription
                }));
                
                // Add message locally for immediate feedback
                addMessage('user', result.transcription);
                setIsProcessing(true);
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

  const finishConversation = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      // Send end_conversation message to backend
      wsRef.current.send(JSON.stringify({
        type: 'end_conversation'
      }));
      
      setIsProcessing(true);
    } else {
      setError('Connection lost. Please refresh and try again.');
    }
  };

  return (
    <Card className={`w-full ${className}`}>
      <CardBody className="p-4">
        <div className="mb-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold flex items-center">
              <span className="mr-2">
                {reminder?.message_type === 'appointment' ? (
                  <ClockIcon className="h-5 w-5 text-blue-500" />
                ) : (
                  <BellIcon className="h-5 w-5 text-purple-500" />
                )}
              </span>
              Appointment Conversation
            </h2>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
            Patient: <span className="font-medium">{reminder?.patient_name}</span> | 
            {reminder?.message_type === 'appointment' ? (
              <span> Doctor: <span className="font-medium">{reminder?.details.doctor_name}</span></span>
            ) : (
              <span> Medication: <span className="font-medium">{reminder?.details.medication_name}</span></span>
            )}
          </p>
        </div>
        {/* Messages */}
        <div className="flex flex-col space-y-4 mb-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[70%] rounded-lg p-3 whitespace-pre-wrap ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : message.role === 'system'
                    ? 'bg-gray-600 text-white'
                    : 'bg-gray-200 text-gray-900'
                }`}
              >
                {message.content}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Controls */}
        <div className="flex flex-col space-y-4 mt-4">
          {/* Text input for testing */}
          <div className="flex items-center space-x-2">
            <input
              type="text"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendTextMessage()}
              placeholder="Type a message..."
              className="flex-1 p-2 border rounded-md"
              disabled={isProcessing || connectionStatus !== 'connected'}
            />
            <Button
              color="primary"
              onClick={sendTextMessage}
              disabled={isProcessing || connectionStatus !== 'connected' || !textInput.trim()}
            >
              Send
            </Button>
          </div>
          
          <div className="flex items-center justify-between">
            <Button
              color={isRecording ? 'danger' : 'primary'}
              onClick={toggleRecording}
              disabled={isProcessing || connectionStatus !== 'connected'}
              className="flex items-center gap-2"
            >
              {isRecording ? (
                <>
                  <StopIcon className="w-5 h-5" />
                  Stop Recording
                </>
              ) : (
                <>
                  <MicrophoneIcon className="w-5 h-5" />
                  Start Recording
                </>
              )}
            </Button>

            {isProcessing && (
              <span className="text-sm text-blue-500 flex items-center">
                <ArrowPathIcon className="w-4 h-4 mr-1 animate-spin" />
                Processing...
              </span>
            )}

            <Button
              color="success"
              onClick={finishConversation}
              disabled={isRecording || isProcessing}
              className="flex items-center gap-2"
            >
              <CheckCircleIcon className="w-5 h-5" />
              Complete
            </Button>
          </div>
        </div>

        {/* Error message */}
        {error && (
          <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        )}
      </CardBody>
    </Card>
  );
}