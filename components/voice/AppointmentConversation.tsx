import React, { useState, useRef, useEffect, forwardRef, useImperativeHandle, useCallback } from 'react';
import { Button } from '@heroui/button';
import { Card, CardBody } from '@heroui/card';
import { ReminderResponse } from '../../app/appointment-manager/types';
import { MicrophoneIcon, StopIcon, SpeakerWaveIcon, CheckCircleIcon, ClockIcon, BellIcon, ArrowPathIcon } from '@heroicons/react/24/solid';
import { appointmentManagerApi } from '../../app/appointment-manager/api';
import { playAudioFromBase64, stopAllAudio } from '../../app/api/audio';
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

export interface AppointmentConversationRef {
  setShouldKeepConnection: (value: boolean) => void;
}

const arrayBufferToBase64 = (buffer: ArrayBuffer): string => {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
};

interface AudioQueueItem {
  audio: string;
  sampleRate: number;
  text?: string;
}

interface SilenceDetectorConfig {
  minSilenceDuration: number;  // milliseconds
  silenceThreshold: number;    // RMS threshold
  maxChunkDuration: number;    // milliseconds
}

interface AudioInputQueue {
  chunks: AudioChunk[];
  isProcessing: boolean;
}

interface AudioOutputQueue {
  chunks: AudioQueueItem[];
  isPlaying: boolean;
}

interface AudioChunk {
  data: Float32Array;
  timestamp: number;
  inProgress?: boolean;
}

class AudioChunkProcessor {
  private buffer: Float32Array[] = [];
  private isInSilence: boolean = false;
  private lastSoundTimestamp: number = Date.now();
  private chunkStartTime: number = Date.now();

  constructor(
    private config: SilenceDetectorConfig = {
      minSilenceDuration: 500,
      silenceThreshold: 0.005,
      maxChunkDuration: 10000
    }
  ) {}

  processSample(sample: Float32Array): { shouldSend: boolean; chunk?: Float32Array } {
    this.buffer.push(sample);
    
    const mav = this.calculateMeanAbsoluteValue(sample);
    console.log('MAV:', mav.toFixed(6));
    const currentTime = Date.now();
    
    if (mav < this.config.silenceThreshold) {
      if (!this.isInSilence) {
        this.isInSilence = true;
        this.lastSoundTimestamp = currentTime;
      }
    } else {
      this.isInSilence = false;
      this.lastSoundTimestamp = currentTime;
    }

    const silenceDuration = currentTime - this.lastSoundTimestamp;
    const chunkDuration = currentTime - this.chunkStartTime;
    
    if (
      (this.isInSilence && silenceDuration >= this.config.minSilenceDuration) ||
      chunkDuration >= this.config.maxChunkDuration
    ) {
      if (this.buffer.length > 0) {
        const chunk = this.concatenateBuffer();
        
        // Only return chunk if it's not completely silent
        if (!this.isChunkSilent(chunk)) {
          this.resetBuffer();
          this.chunkStartTime = currentTime;
          return { shouldSend: true, chunk };
        }
        
        // If chunk is silent, just reset buffer without sending
        this.resetBuffer();
        this.chunkStartTime = currentTime;
      }
    }

    return { shouldSend: false };
  }

  private calculateMeanAbsoluteValue(samples: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < samples.length; i++) {
      sum += Math.abs(samples[i]);
    }
    return sum / samples.length;
  }

  private isChunkSilent(chunk: Float32Array): boolean {
    const mav = this.calculateMeanAbsoluteValue(chunk);
    return mav < this.config.silenceThreshold;
  }

  private concatenateBuffer(): Float32Array {
    const totalLength = this.buffer.reduce((acc, curr) => acc + curr.length, 0);
    const result = new Float32Array(totalLength);
    let offset = 0;
    
    for (const chunk of this.buffer) {
      result.set(chunk, offset);
      offset += chunk.length;
    }
    
    return result;
  }

  private resetBuffer(): void {
    this.buffer = [];
  }
}


export const AppointmentConversation = forwardRef<AppointmentConversationRef, ConversationPanelProps>(
  function AppointmentConversation({ reminder, onComplete, className = "" }, ref) {
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
  const [ttsPlaying, setTtsPlaying] = useState(false);
  const [audioQueue, setAudioQueue] = useState<AudioQueueItem[]>([]);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const audioQueueRef = useRef<AudioQueueItem[]>([]);
  
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
  const connectionTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const transcriptionBufferRef = useRef<string>(''); // Add transcription buffer ref
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  
  // Add these refs after other refs
  const inputQueueRef = useRef<AudioInputQueue>({
    chunks: [],
    isProcessing: false
  });

  const outputQueueRef = useRef<AudioOutputQueue>({
    chunks: [],
    isPlaying: false
  });

  // Add constants for connection management
  const PING_INTERVAL = 15000; // 15 seconds
  const MAX_RECONNECT_ATTEMPTS = 5;
  const RECONNECT_DELAY = 2000; // 2 seconds
  
  // Add new state for conversation flow
  const [conversationState, setConversationState] = useState<'initial' | 'context' | 'confirmation'>('initial');
  
  // Add to existing state declarations
  const [appointmentDetails, setAppointmentDetails] = useState<any>(null);

  // Add text input functionality for testing
  const [textInput, setTextInput] = useState<string>('');

  // Add a new state to track if we should keep the connection alive
  const [shouldKeepConnection, setShouldKeepConnection] = useState(true);

  // Add this state at the top of the component with other states
  const [currentTranscription, setCurrentTranscription] = useState<string>('');

  // Expose the setShouldKeepConnection method via ref with audio stopping
  useImperativeHandle(ref, () => ({
    setShouldKeepConnection: (value: boolean) => {
      console.log(`[AppointmentConversation] Setting shouldKeepConnection to ${value}`);
      setShouldKeepConnection(value);
      
      // If setting to false, immediately close the WebSocket
      if (value === false && wsRef.current) {
        // If the connection is open, send a cleanup message before closing
        if (wsRef.current.readyState === WebSocket.OPEN) {
          console.log('[AppointmentConversation] Sending cleanup message before closing connection');
          wsRef.current.send(JSON.stringify({
            type: 'client_cleanup',
            reason: 'navigation'
          }));
          
          // Short timeout to allow the message to be sent before closing
          setTimeout(() => {
            console.log('[AppointmentConversation] Closing WebSocket connection after cleanup message');
            wsRef.current?.close();
            
            // Also clear all intervals and timeouts
            if (reconnectTimeoutRef.current) {
              clearTimeout(reconnectTimeoutRef.current);
              reconnectTimeoutRef.current = null;
            }
            if (pingIntervalRef.current) {
              clearInterval(pingIntervalRef.current);
              pingIntervalRef.current = null;
            }
            if (connectionTimeoutRef.current) {
              clearTimeout(connectionTimeoutRef.current);
              connectionTimeoutRef.current = null;
            }
            
            // Reset TTS playing state and stop any audio
            setTtsPlaying(false);
            stopAllAudio();
          }, 1);
        } else {
          // Connection is already closed or closing
          console.log('[AppointmentConversation] Connection not open, skipping cleanup message');
          
          // Clear all intervals and timeouts
          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
          }
          if (pingIntervalRef.current) {
            clearInterval(pingIntervalRef.current);
            pingIntervalRef.current = null;
          }
          if (connectionTimeoutRef.current) {
            clearTimeout(connectionTimeoutRef.current);
            connectionTimeoutRef.current = null;
          }
          
          // Reset TTS playing state and stop any audio
          setTtsPlaying(false);
          stopAllAudio();
        }
      }
    }
  }));

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
        if (!shouldKeepConnection && wsRef.current) {
          console.log('[WebSocket] Component unmounting, closing connection');
          wsRef.current.close();
        }
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
        }
        setTtsPlaying(false);
        
        // Stop any playing audio when unmounting
        stopAllAudio();
      };
    }
    
    return () => {
      // Ensure audio is stopped even if reminder is undefined
      stopAllAudio();
    };
  }, [reminder, shouldKeepConnection]);

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
    if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
      const delay = RECONNECT_DELAY * Math.pow(2, reconnectAttemptsRef.current);
      console.log(`[WebSocket] Scheduling reconnection attempt ${reconnectAttemptsRef.current + 1}/${MAX_RECONNECT_ATTEMPTS} in ${delay}ms`);
      
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log(`[WebSocket] Executing reconnection attempt ${reconnectAttemptsRef.current + 1}`);
        reconnectAttemptsRef.current += 1;
        connectWebSocket(patientId);
      }, delay);
    } else {
      console.log('[WebSocket] Maximum reconnection attempts reached');
      setError('Failed to reconnect after multiple attempts. Please refresh the page.');
    }
  };

  const connectWebSocket = async (patientId: string) => {
    try {
      console.log('[WebSocket] Starting connection process...');
      setConnectionStatus('connecting');
      
      // Create WebSocket connection to our appointment endpoint
      const reminderId = reminder?.id || 'default';
      const wsUrl = `${process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'}/ws/${reminderId}`;
      
      console.log(`[WebSocket] Connecting to: ${wsUrl}`);
      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        console.log('[WebSocket] Connection opened successfully');
        setConnectionStatus('connected');
        setError(null);
        reconnectAttemptsRef.current = 0;
        
        // Send initial context to the WebSocket
        if (reminder) {
          console.log('[WebSocket] Sending initial context:', JSON.stringify(reminder, null, 2));
          if (wsRef.current) {
            wsRef.current.send(JSON.stringify({
              type: 'context',
              context: JSON.stringify(reminder)
            }));
          }
        }
        
        // Set up client-side ping interval
        if (pingIntervalRef.current) {
          console.log('[WebSocket] Clearing existing ping interval');
          clearInterval(pingIntervalRef.current);
        }
        
        console.log('[WebSocket] Setting up new ping interval');
        pingIntervalRef.current = setInterval(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            console.debug('[WebSocket] Sending ping message');
            wsRef.current.send(JSON.stringify({ type: 'ping' }));
          } else {
            console.log('[WebSocket] Cannot send ping - connection not open');
          }
        }, PING_INTERVAL);
      };
      
      wsRef.current.onmessage = async (event) => {
        const data = JSON.parse(event.data);
        console.log('[WebSocket] Received message:', JSON.stringify(data, null, 2));
        
        switch (data.type) {
          case 'message':
            console.log('[WebSocket] Processing message:', {
              role: data.role || 'agent',
              text: data.text,
              isGoodbye: data.is_goodbye
            });
            addMessage(data.role || 'agent', data.text);

            
            if (data.is_goodbye) {
              console.log('[WebSocket] Received goodbye message, setting processing state');
              setIsProcessing(true);
            } else {
              setIsProcessing(false);
            }
            break;
            
          case 'ping':
            console.debug('[WebSocket] Received ping from server');
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              console.debug('[WebSocket] Sending pong response');
              wsRef.current.send(JSON.stringify({ type: 'pong' }));
            } else {
              console.log('[WebSocket] Cannot send pong - connection not open');
            }
            lastPongTimeRef.current = Date.now();
            break;
            
          case 'pong':
            console.debug('[WebSocket] Received pong from server');
            lastPongTimeRef.current = Date.now();
            break;
            
          case 'appointment_result':
            console.log('[WebSocket] Processing appointment result:', JSON.stringify(data, null, 2));
            if (data.success && data.appointment) {
              const appointment = data.appointment;
              console.log('[WebSocket] Setting appointment details:', appointment);
              setAppointmentDetails({
                ...appointment,
                status: appointment.status || 'confirmed'
              });
              if (onComplete) {
                console.log('[WebSocket] Calling onComplete with success');
                onComplete({
                  status: appointment.status || 'confirmed',
                  appointment: appointment,
                  success: true
                });
              }
            } else {
              const errorMsg = data.message || 'Failed to process appointment';
              console.log('[WebSocket] Appointment processing failed:', errorMsg);
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
            console.log('[WebSocket] Processing summary:', data);
            if (onComplete) {
              onComplete(data.details || appointmentDetails || { status: 'completed' });
            }
            break;
            
          case 'error':
            console.log('[WebSocket] Received error:', data.message);
            setError(data.message || 'An error occurred');
            setIsProcessing(false);
            break;
          
            
          case 'audio':
            console.log('[WebSocket] Received audio data from server', {
              dataLength: data.audio ? data.audio.length : 0,
              sampleRate: data.sample_rate,
              text: data.text ? data.text.substring(0, 30) + '...' : null
            });
            
            if (!data.audio) {
              console.error('[WebSocket] Received audio message with no audio data');
              break;
            }

            // Add the new chunk to the queue
            setAudioQueue(prev => [...prev, {
              audio: data.audio,
              sampleRate: data.sample_rate || 24000,
              text: data.text
            }]);
            break;
            
          default:
            console.log('[WebSocket] Unhandled message type:', data.type);
            break;
        }
      };
      
      wsRef.current.onclose = (event) => {
        console.log('[WebSocket] Connection closed:', {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean,
          currentStatus: connectionStatus
        });
        if (connectionStatus === 'connected') {
          console.log('[WebSocket] Connection was previously connected, attempting reconnection');
          setConnectionStatus('disconnected');
          handleReconnection(patientId);
        }
        
        if (pingIntervalRef.current) {
          console.log('[WebSocket] Clearing ping interval');
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }
      };
      
      wsRef.current.onerror = (error) => {
        console.error('[WebSocket] Connection error:', error);
        setConnectionStatus('error');
        setError('WebSocket connection error');
      };
    } catch (error) {
      console.error('[WebSocket] Error in connection process:', error);
      setConnectionStatus('error');
      setError('Failed to establish connection');
    }
  };

const processNextChunk = async () => {
    if (inputQueueRef.current.isProcessing || inputQueueRef.current.chunks.length === 0) {
        return;
    }

    inputQueueRef.current.isProcessing = true;
    const chunk = inputQueueRef.current.chunks[0];

    try {
        const uint8Array = new Uint8Array(chunk.data.buffer);
        let base64Data = '';
        for (let i = 0; i < uint8Array.length; i++) {
            base64Data += String.fromCharCode(uint8Array[i]);
        }
        base64Data = btoa(base64Data);

        const baseUrl = process.env.NEXT_PUBLIC_STT_SERVER_URL?.replace(/\/+$/, '');
        const transcribeUrl = `${baseUrl}/transcribe`;

        const response = await fetch(transcribeUrl, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ audio: base64Data })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        if (result.transcription?.trim()) {
            transcriptionBufferRef.current += ' ' + result.transcription.trim();
            setCurrentTranscription(transcriptionBufferRef.current);
        }

    } catch (error) {
        console.error("Error processing audio chunk:", error);
    } finally {
        // Remove the processed chunk and reset flag
        inputQueueRef.current.chunks.shift();
        inputQueueRef.current.isProcessing = false;
        
        // Process next chunk if available
        if (inputQueueRef.current.chunks.length > 0) {
            processNextChunk();
        }
    }
};

const handleStartRecording = async () => {
    try {
        if (isRecording) {
            await handleStopRecording();
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            setError('Connection not established');
            return;
        }

        // Clear previous state
        transcriptionBufferRef.current = '';
        setCurrentTranscription('');
        inputQueueRef.current.chunks = [];
        inputQueueRef.current.isProcessing = false;

        // Get microphone stream
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
        
        // Create and store audio processing nodes
        audioContextRef.current = new AudioContext({ sampleRate: 16000 });
        sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
        processorRef.current = audioContextRef.current.createScriptProcessor(1024, 1, 1);
        
        // Initialize audio chunk processor
        const chunkProcessor = new AudioChunkProcessor({
            minSilenceDuration: 500,
            silenceThreshold: 0.005,
            maxChunkDuration: 10000
        });

        // Connect audio nodes
        sourceRef.current.connect(processorRef.current);
        processorRef.current.connect(audioContextRef.current.destination);
        
        // Set up audio processing
        processorRef.current.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            const { shouldSend, chunk } = chunkProcessor.processSample(new Float32Array(inputData));
            
            if (shouldSend && chunk) {
                // Add to input queue instead of processing immediately
                inputQueueRef.current.chunks.push({
                    data: chunk,
                    timestamp: Date.now()
                });
                
                // Trigger processing if not already in progress
                if (!inputQueueRef.current.isProcessing) {
                    processNextChunk();
                }
            }
        };

        setIsRecording(true);
        
    } catch (error: any) {
        console.error("Error starting recording:", error);
        await handleStopRecording();
        
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

const handleStopRecording = async () => {
    try {
        // 1. Stop processing first
        if (processorRef.current) {
            processorRef.current.onaudioprocess = null; // Remove the event handler first
            processorRef.current.disconnect();
            processorRef.current = null;
        }

        // 2. Disconnect source
        if (sourceRef.current) {
            sourceRef.current.disconnect();
            sourceRef.current = null;
        }

        // 3. Close audio context
        if (audioContextRef.current) {
            await audioContextRef.current.close();
            audioContextRef.current = null;
        }

        // 4. Stop all tracks in the stream
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => {
                track.stop();
            });
            streamRef.current = null;
        }

        // 5. Clear input queue
        inputQueueRef.current = {
            chunks: [],
            isProcessing: false
        };
        
        // Send any remaining transcription
        if (transcriptionBufferRef.current && transcriptionBufferRef.current.trim()) {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({
                    type: 'message',
                    text: transcriptionBufferRef.current.trim()
                }));
                addMessage('user', transcriptionBufferRef.current.trim());
            }
        }
        
        // Clear transcription buffers
        transcriptionBufferRef.current = '';
        setCurrentTranscription('');
        
        // Update recording state
        setIsRecording(false);
        
    } catch (error) {
        console.error("Error cleaning up recording resources:", error);
        // Still try to update state even if cleanup failed
        setIsRecording(false);
    }
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
      
      // Stop any playing audio
      stopAllAudio();
    } else {
      setError('Connection lost. Please refresh and try again.');
    }
  };

  // Modify the onComplete handler to set shouldKeepConnection to false
  const handleCompleteConversation = (result: ConversationResult) => {
    if (!result) {
      setError("No result was produced. The conversation may have failed.");
      setConversationState('initial');
      return;
    }
    
    setAppointmentDetails(result);
    setConversationState('confirmation');
    setShouldKeepConnection(false); // Signal that we're done with the conversation
    
    // Stop any playing audio
    stopAllAudio();
  };

  // Add a cleanup effect just for audio
  useEffect(() => {
    return () => {
      // Stop any audio that might be playing when component unmounts
      stopAllAudio();
    };
  }, []);

  // Keep audioQueueRef in sync with audioQueue state
  useEffect(() => {
    audioQueueRef.current = audioQueue;
  }, [audioQueue]);

  const processAudioQueue = useCallback(async () => {
    if (isPlayingAudio || audioQueueRef.current.length === 0) {
      return;
    }

    console.log('[WebSocket] TTS playback started');
    setIsPlayingAudio(true);
    const nextChunk = audioQueueRef.current[0];

    try {
      await playAudioFromBase64(
        nextChunk.audio,
        nextChunk.sampleRate || 24000,
        'PCM_16'
      );
      
      // Remove the played chunk from queue
      setAudioQueue(prev => prev.slice(1));

      if (audioQueue.length === 0) {
        setTtsPlaying(false);
      }
    } catch (error) {
      console.error('[Audio] Error playing audio chunk:', error);
    } finally {
      setIsPlayingAudio(false);
      console.log('[WebSocket] TTS playback completed');
    }
  }, []);

  // Monitor queue and process chunks
  useEffect(() => {
    if (!isPlayingAudio && audioQueue.length > 0) {
      setTtsPlaying(true);
      processAudioQueue();
    }
  }, [audioQueue, isPlayingAudio, processAudioQueue]);

  // Add cleanup for audio queue when component unmounts
  useEffect(() => {
    return () => {
      setAudioQueue([]);
      setIsPlayingAudio(false);
      stopAllAudio();
    };
  }, []);

  // Add this useEffect for microphone cleanup
  useEffect(() => {
    return () => {
      // Cleanup microphone on unmount
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          track.stop();
        });
        streamRef.current = null;
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
        mediaRecorderRef.current = null;
      }
      setIsRecording(false);
    };
  }, []);

  useEffect(() => {
    return () => {
      // Audio cleanup
      stopAllAudio();
      setAudioQueue([]);
      setIsPlayingAudio(false);
      setTtsPlaying(false);
      
      // Microphone cleanup
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
        mediaRecorderRef.current = null;
      }
      setIsRecording(false);
      
      // WebSocket cleanup
      if (!shouldKeepConnection && wsRef.current) {
        wsRef.current.close();
      }
      
      // Clear all intervals and timeouts
      [reconnectTimeoutRef, pingIntervalRef, connectionTimeoutRef].forEach(ref => {
        if (ref.current) {
          clearTimeout(ref.current);
          ref.current = null;
        }
      });

      // Clear transcription
      transcriptionBufferRef.current = '';
      setCurrentTranscription('');
    };
  }, [shouldKeepConnection]);

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
        <div className="flex flex-col space-y-4 mb-4 relative">
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
          
          {ttsPlaying && (
            <div className="absolute bottom-0 right-0 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-3 py-1 rounded-full flex items-center shadow-md z-10">
              <SpeakerWaveIcon className="w-4 h-4 mr-1 animate-pulse" />
              <span className="text-xs font-medium">Speaking...</span>
            </div>
          )}
        </div>

        {/* Current transcription display */}
        {isRecording && currentTranscription && (
          <div className="mb-4 px-4 py-2 bg-gray-100 dark:bg-gray-800 rounded-lg">
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
              Currently transcribing:
            </div>
            <div className="text-gray-900 dark:text-gray-100">
              {currentTranscription}
            </div>
          </div>
        )}

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
});