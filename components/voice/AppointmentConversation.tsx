import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@heroui/button';
import { Card, CardBody } from '@heroui/card';
import { ReminderResponse } from '../../app/appointment-manager/types';
import { MicrophoneIcon, StopIcon, SpeakerWaveIcon, CheckCircleIcon } from '@heroicons/react/24/solid';
import { appointmentManagerApi } from '../../app/appointment-manager/api';
import { playAudioFromBase64 } from '../../app/api/audio';
import { Input } from '@heroui/input';


interface Message {
  role: 'user' | 'system' | 'agent';
  content: string;
  timestamp: Date;
  audio?: string; // Base64 audio data for playback
  sampleRate?: number;
}

interface ConversationPanelProps {
  reminder: ReminderResponse;
  onComplete: () => void;
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
  const [availableSlots, setAvailableSlots] = useState<any[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  const [error, setError] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [audioData, setAudioData] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioBufferRef = useRef<Float32Array[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const latestAudioRef = useRef<string | null>(null);

  // Add new state for conversation flow
  const [conversationState, setConversationState] = useState<'initial' | 'context' | 'specialty_search' | 'doctor_selection' | 'appointment_scheduling' | 'confirmation'>('initial');
  const [isListening, setIsListening] = useState(false);
  const [specialtySearchComplete, setSpecialtySearchComplete] = useState(false);
  
  // Add to existing state declarations
  const [selectedDoctor, setSelectedDoctor] = useState<any>(null);
  const [appointmentDetails, setAppointmentDetails] = useState<any>(null);

  // Connect to the conversation when a reminder is selected
  useEffect(() => {
    if (reminder) {
      console.log("Setting up conversation with reminder:", reminder);
      const id = `${reminder.message_type}_${reminder.details.patient_id || 'unknown'}_${Date.now()}`;
      setConversationId(id);
      
      // Force the initial state
      setConversationState('initial');
      console.log("Set conversation state to initial");
      
      connectWebSocket(reminder.details.patient_id?.toString() || '1');
      
      // Explicitly set the conversation state to initial
      setConversationState('initial');
      
      // Add welcome message
      const welcomeMessage = `Hello! I'm your appointment assistant. You have an appointment with Dr. ${reminder.details.doctor_name} on ${new Date(reminder.details.datetime).toLocaleString()}.`;
      
      setMessages([{ 
        role: 'system', 
        content: welcomeMessage, 
        timestamp: new Date() 
      }]);
    }
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
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

  // Fetch audio data from the backend for this conversation
  const fetchConversationAudio = async (id: string) => {
    try {
      const response = await fetch(`/api/audio-data?sessionId=${id}`);
      if (response.ok) {
        const data = await response.json();
        if (data.audioData) {
          setAudioData(data.audioData);
        }
      }
    } catch (error) {
      console.error("Error fetching conversation audio:", error);
    }
  };

  // Audio processing setup
  const setupAudioProcessing = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new AudioContext();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);

      source.connect(processor);
      processor.connect(audioContextRef.current.destination);

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        audioBufferRef.current.push(new Float32Array(inputData));
      };

      // Setup MediaRecorder for chunked recording
      mediaRecorderRef.current = new MediaRecorder(stream);
      mediaRecorderRef.current.ondataavailable = handleAudioData;
      
      return stream;
    } catch (error) {
      console.error('Error setting up audio:', error);
      setError('Failed to setup audio recording');
      return null;
    }
  };

  // Handle audio data chunks
  const handleAudioData = async (event: BlobEvent) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    try {
      const arrayBuffer = await event.data.arrayBuffer();
      const float32Array = new Float32Array(arrayBuffer);
      const base64Audio = arrayBufferToBase64(float32Array.buffer);

      wsRef.current.send(JSON.stringify({
        type: 'audio_input',
        audio: base64Audio,
        timestamp: Date.now()
      }));
    } catch (error) {
      console.error('Error processing audio data:', error);
    }
  };

  // Add this before handleWebSocketMessage
  const handleConversationSummary = (data: { summary: string }) => {
    if (data.summary) {
      addMessage('agent', data.summary);
      setConversationState('confirmation');
    }
  };

  // Then the existing handleWebSocketMessage function
  const handleWebSocketMessage = async (event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);
      console.log('WebSocket message received:', data);

      switch (data.type) {
        case 'message':
          addMessage('agent', data.text);
          if (data.audio) {
            await playAudioResponse(data.audio, data.sample_rate);
          }
          
          // Handle conversation state transitions based on message content
          if (data.text.includes('Finding a suitable doctor')) {
            setConversationState('specialty_search');
          } else if (data.text.includes('available slots')) {
            setConversationState('doctor_selection');
          } else if (data.text.includes('appointment confirmed')) {
            setConversationState('confirmation');
          }
          break;

        case 'audio':
          // TTS audio responses
          await playAudioResponse(data.audio, data.sample_rate);
          break;

        case 'transcription':
          // Speech-to-text results
          addMessage('user', data.text);
          break;

        case 'doctors':
          // Doctor availability information
          handleDoctorsList(data.data);
          setSpecialtySearchComplete(true);
          break;

        case 'appointment':
          setAppointmentDetails(data.details);
          break;

        case 'error':
          setError(data.message || 'An error occurred');
          break;

        case 'summary':
          handleConversationSummary({ summary: data.text });
          break;

        default:
          console.warn('Unknown message type:', data.type);
      }
    } catch (err) {
      console.error('Error handling WebSocket message:', err);
    }
  };

  // Modified WebSocket connection setup
  const connectWebSocket = (patientId: string) => {
    if (!reminder) return;

    setConnectionStatus('connecting');
    // Add debug parameter to URL
    const wsUrl = `${process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'}/ws/schedule/${patientId}?debug=true`;
    
    try {
      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        console.log('WebSocket connection established');
        // Send initial debug message
        wsRef.current?.send(JSON.stringify({
          type: 'debug',
          message: 'Client connected and ready',
          client_info: {
            reminder_type: reminder.message_type,
            patient_id: reminder.details.patient_id
          }
        }));
        setConnectionStatus('connected');
        setError(null);
      };

      wsRef.current.onerror = (error) => {
        // Send error info to backend
        wsRef.current?.send(JSON.stringify({
          type: 'debug',
          message: 'WebSocket error on client',
          error: error.toString()
        }));
        setConnectionStatus('error');
        setError('Failed to connect to scheduling service. Please try again.');
      };

      wsRef.current.onmessage = handleWebSocketMessage;

    } catch (error) {
      console.error('Error creating WebSocket:', error);
      setConnectionStatus('error');
      setError('Failed to establish connection');
    }
  };

  // Helper functions
  const arrayBufferToBase64 = (buffer: ArrayBuffer): string => {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  };

  const playAudioResponse = async (base64Audio: string, sampleRate: number = 24000) => {
    try {
      const audioContext = new AudioContext();
      const audioData = atob(base64Audio);
      const arrayBuffer = new ArrayBuffer(audioData.length);
      const view = new Uint8Array(arrayBuffer);
      
      for (let i = 0; i < audioData.length; i++) {
        view[i] = audioData.charCodeAt(i);
      }

      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);
      source.start(0);
    } catch (error) {
      console.error('Error playing audio response:', error);
    }
  };

  const handleDoctorsList = (doctors: any[]) => {
    const doctorsMessage = doctors
      .map(doc => `${doc.doctor_name} (${doc.specialty})\nAvailable slots:\n${doc.available_slots.join('\n')}`)
      .join('\n\n');
    addMessage('agent', doctorsMessage);
  };

  const handleReconnection = (patientId: string) => {
    if (reconnectAttemptsRef.current < 5) {
      setTimeout(() => {
        reconnectAttemptsRef.current += 1;
        connectWebSocket(patientId);
      }, 2000 * Math.pow(2, reconnectAttemptsRef.current));
    }
  };

  // Cleanup function
  const cleanup = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
  };

  // Effect for cleanup
  useEffect(() => {
    return cleanup;
  }, []);

  // Modified handleContextSubmit to match scheduler.py flow
  const handleContextSubmit = () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Connection not established');
      return;
    }
    
    wsRef.current.send(JSON.stringify({
      type: 'context',
      context: context,
      patient_id: reminder?.details.patient_id
    }));
    
    setIsContextSent(true);
    setConversationState('specialty_search');
    startListening(); // Start listening for voice input after context is sent
  };

  const addMessage = (role: 'user' | 'system' | 'agent', content: string) => {
    setMessages(prevMessages => [
      ...prevMessages, 
      { role, content, timestamp: new Date() }
    ]);

    if (role === 'agent' || role === 'system') {
      setIsProcessing(false);
    }
  };

  const saveAudioData = async (sessionId: string, audioData: string) => {
    try {
      await fetch('/api/audio-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ sessionId, audioData })
      });
    } catch (error) {
      console.error('Error saving audio data:', error);
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const startRecording = () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Connection not established. Please try again.');
      return;
    }

    setIsRecording(true);
    setError(null);
    wsRef.current.send(JSON.stringify({ command: 'start_listening' }));
  };

  const stopRecording = () => {
    setIsRecording(false);
    setIsProcessing(true);
    
    // Add a slight delay to finish processing any remaining audio
    setTimeout(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        // Optionally send a stop command to the server
        wsRef.current.send(JSON.stringify({ command: 'stop_listening' }));
      }
    }, 500);
  };

  const finishConversation = async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Connection not established. Please try again.');
      return;
    }

    setIsProcessing(true);
    
    // Let the backend know we're done
    wsRef.current.send(JSON.stringify({ command: 'finish' }));
    
    // If we have a reminder with backend conversation audio, try to fetch it
    if (reminder && conversationId) {
      try {
        const response = await fetch(`/api/get-conversation-audio?conversation_id=${conversationId}`);
        
        if (response.ok) {
          const data = await response.json();
          if (data.audio_data) {
            await saveAudioData(conversationId, data.audio_data);
            setAudioData(data.audio_data);
          }
        }
      } catch (error) {
        console.error('Error fetching conversation audio from backend:', error);
      }
    }
    
    // Let the parent component know we're done
    if (onComplete) {
      const result: {
        status: string;
        type: string;
        patient_name: string;
        datetime: Date;
        audioAvailable: boolean;
        audioData: string | null;
        doctor_name?: string;
        appointment_datetime?: string;
        medication_name?: string;
        dosage?: string;
        adherence?: string;
      } = {
        status: 'completed',
        type: reminder?.message_type || '',
        patient_name: reminder?.details?.patient_name || '',
        datetime: new Date(),
        audioAvailable: !!audioData,
        audioData: audioData
      };
      
      if (reminder?.message_type === 'appointment') {
        result.doctor_name = reminder.details.doctor_name;
        result.appointment_datetime = reminder.details.datetime;
      } else if (reminder?.message_type === 'medication') {
        result.medication_name = reminder.details.medication_name;
        result.dosage = reminder.details.dosage;
        result.adherence = "Confirmed";
      }
      
      onComplete();
    }
  };

  const playAudio = async (base64Audio: string, sampleRate: number = 16000) => {
    try {
      setIsSpeaking(true);
      await playAudioFromBase64(base64Audio);
      setIsSpeaking(false);
    } catch (error) {
      console.error('Error playing audio:', error);
      setIsSpeaking(false);
    }
  };

  const replayAudio = async (message: Message) => {
    if (message.audio) {
      await playAudio(message.audio, message.sampleRate);
    } else if (audioData && conversationId) {
      // If we have audio data cached for this conversation, try to play it
      await playAudio(audioData);
    }
  };

  // Get appropriate connection status indicator
  const getConnectionStatusIndicator = () => {
    switch (connectionStatus) {
      case 'connected':
        return <span className="text-xs bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 px-2 py-1 rounded-full flex items-center">
          <span className="w-2 h-2 bg-green-500 rounded-full mr-1"></span>
          Connected
        </span>;
      case 'connecting':
        return <span className="text-xs bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200 px-2 py-1 rounded-full flex items-center">
          <span className="w-2 h-2 bg-yellow-500 rounded-full mr-1 animate-pulse"></span>
          Connecting...
        </span>;
      case 'disconnected':
        return <span className="text-xs bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 px-2 py-1 rounded-full flex items-center">
          <span className="w-2 h-2 bg-red-500 rounded-full mr-1"></span>
          Disconnected
        </span>;
      case 'error':
        return <span className="text-xs bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 px-2 py-1 rounded-full flex items-center">
          <span className="w-2 h-2 bg-red-500 rounded-full mr-1"></span>
          Connection Error
        </span>;
    }
  };

  // Add function to start voice input listening
  const startListening = async () => {
    setIsListening(true);
    const stream = await setupAudioProcessing();
    if (stream && mediaRecorderRef.current) {
      mediaRecorderRef.current.start(1000); // Start recording in 1-second chunks
    }
  };

  // Add function to stop listening
  const stopListening = () => {
    setIsListening(false);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
  };

  // Add conversation state effect
  useEffect(() => {
    if (conversationState === 'confirmation') {
      cleanup();
      onComplete && onComplete();
    }
  }, [conversationState]);

  console.log("Current conversation state:", conversationState);

  return (
    <Card className={`w-full ${className}`}>
      <CardBody>
        {/* Debug Button */}
        <Button
          color="primary"
          className="mb-4"
          onClick={() => {
            wsRef.current?.send(JSON.stringify({
              type: 'debug',
              message: 'Debug button clicked',
              timestamp: new Date().toISOString()
            }));
          }}
        >
          Send Debug Message
        </Button>

        {/* Initial Context Stage */}
        {conversationState === 'initial' && (
          <div className="flex flex-col space-y-4">
            <h2 className="text-xl font-semibold text-white">Start Scheduling Assistant</h2>
            <div className="bg-gray-800 rounded-lg p-6">
              <p className="text-gray-400 mb-4">
                Please provide any specific requirements or preferences for your appointment:
              </p>
              <Input
                as="textarea"
                className="w-full p-3 bg-gray-900 border border-gray-700 rounded-lg text-gray-200 mb-4"
                value={context}
                onChange={(e) => setContext(e.target.value)}
                placeholder="E.g., I need an appointment with a cardiologist, preferably in the morning..."
                rows={4}
              />
              <Button
                color="primary"
                className="w-full"
                onClick={handleContextSubmit}
                disabled={!context.trim() || connectionStatus !== 'connected'}
              >
                Start Scheduling
              </Button>
            </div>
          </div>
        )}

        {/* Recording and Transcription Stage */}
        {conversationState !== 'initial' && (
          <div className="flex flex-col space-y-4">
            <div className="grid grid-cols-2 gap-4">
              {/* Voice Input Panel */}
              <div className="bg-gray-800 rounded-lg p-6">
                <h2 className="text-xl font-semibold mb-4 text-white">Voice Input</h2>
                <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 mb-4 flex flex-col items-center justify-center text-gray-400">
                  <p className="text-center mb-2">Click Start Recording to begin</p>
                  <p className="text-sm text-center">Audio will be processed when you stop</p>
                </div>
                <Button
                  color="primary"
                  className="w-full flex items-center justify-center space-x-2"
                  onClick={toggleRecording}
                  disabled={!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN}
                >
                  {isRecording ? (
                    <StopIcon className="h-5 w-5" />
                  ) : (
                    <MicrophoneIcon className="h-5 w-5" />
                  )}
                  <span>{isRecording ? 'Stop Recording' : 'Start Recording'}</span>
                </Button>
              </div>

              {/* Transcription Panel */}
              <div className="bg-gray-800 rounded-lg p-6">
                <h2 className="text-xl font-semibold mb-4 text-white">Transcription</h2>
                <div className="bg-gray-900 rounded-lg p-4 h-[200px] overflow-y-auto">
                  <p className="text-gray-400">
                    {messages.length > 0 
                      ? messages[messages.length - 1].content 
                      : 'Transcribed text will appear here...'}
                  </p>
                </div>
              </div>
            </div>

            {/* Schedule Appointment Button */}
            <div className="flex justify-end">
              <Button
                color="primary"
                onClick={finishConversation}
                disabled={isProcessing || messages.length === 0}
              >
                Schedule Appointment
              </Button>
            </div>
          </div>
        )}

        {/* Connection Status */}
        <div className="absolute top-4 right-4">
          {getConnectionStatusIndicator()}
        </div>

        {/* Error Message */}
        {error && (
          <div className="absolute bottom-4 left-4 right-4">
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative">
              {error}
            </div>
          </div>
        )}
      </CardBody>
    </Card>
  );
}