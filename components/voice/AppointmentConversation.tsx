import { useState, useEffect, useRef } from 'react';
import { Button } from '@heroui/button';
import { Card, CardBody } from '@heroui/card';
import { ReminderResponse } from '../../app/appointment-manager/types';
import { MicrophoneIcon, StopIcon, SpeakerWaveIcon, CheckCircleIcon } from '@heroicons/react/24/solid';
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

interface ConversationPanelProps {
  reminder?: ReminderResponse;
  onComplete?: (result: any) => void;
  className?: string;
}

export function AppointmentConversation({ reminder, onComplete, className = "" }: ConversationPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [availableSlots, setAvailableSlots] = useState<any[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('connecting');
  const [error, setError] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [audioData, setAudioData] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const latestAudioRef = useRef<string | null>(null);

  // Connect to the conversation when a reminder is selected
  useEffect(() => {
    if (reminder) {
      // Create a unique ID for this conversation based on reminder details
      const id = `${reminder.message_type}_${reminder.details.patient_id || 'unknown'}_${Date.now()}`;
      setConversationId(id);
      connectWebSocket(reminder.details.patient_id?.toString() || '1');
      
      // Add welcome message
      const welcomeMessage = reminder.message_type === 'appointment'
        ? `Hello! I'm your appointment assistant. You have an appointment with Dr. ${reminder.details.doctor_name} on ${new Date(reminder.details.datetime).toLocaleString()}.`
        : `Hello! I'm your medication assistant. This is a reminder about your medication: ${reminder.details.medication_name}, dosage: ${reminder.details.dosage}, frequency: ${reminder.details.frequency}.`;
      
      setMessages([{ 
        role: 'system', 
        content: welcomeMessage, 
        timestamp: new Date() 
      }]);
      
      // Fetch audio for this conversation if available
      fetchConversationAudio(id);
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

  const connectWebSocket = (patientId: string) => {
    if (!reminder) return;

    setConnectionStatus('connecting');
    
    // Use consistent API configuration for WebSocket connection
    const wsUrl = `${process.env.NEXT_PUBLIC_SPEECH_SERVICE_URL || 'ws://localhost:8000'}/ws/schedule/${patientId}?token=${API_CONFIG.websocketAuth.token}`;
    console.log(`Connecting to WebSocket at ${wsUrl}`);
    
    try {
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connection established');
        setConnectionStatus('connected');
        reconnectAttemptsRef.current = 0; // Reset reconnect attempts
        
        // Send initial context about the appointment
        const initialContext = reminder.message_type === 'appointment'
          ? `Appointment reminder for ${reminder.patient_name} with Dr. ${reminder.details.doctor_name} at ${new Date(reminder.details.datetime).toLocaleString()}.`
          : `Medication reminder for ${reminder.patient_name}: ${reminder.details.medication_name}, dosage: ${reminder.details.dosage}, frequency: ${reminder.details.frequency}.`;
        
        ws.send(JSON.stringify({
          context: initialContext
        }));
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('WebSocket message received:', data);
          
          if (data.type === 'message') {
            addMessage('agent', data.text);
          } 
          else if (data.type === 'audio') {
            // Store audio data for playback
            const message = messages[messages.length - 1];
            if (message && message.role === 'agent') {
              message.audio = data.audio;
              message.sampleRate = data.sample_rate || 16000;
              setMessages([...messages]);
              
              // Play audio immediately
              playAudio(data.audio, data.sample_rate || 16000, data.encoding || 'PCM_FLOAT');
            }
          }
          else if (data.type === 'transcription') {
            addMessage('user', data.text);
          }
          else if (data.type === 'error') {
            console.error('WebSocket error:', data.message);
            setError(data.message);
          }
          else if (data.type === 'status') {
            console.log('Status update:', data.message);
            // Handle status updates like "waiting_for_input"
            if (data.status === 'waiting_for_input') {
              setIsWaitingForInput(true);
            }
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      ws.onclose = (event) => {
        console.log('WebSocket connection closed', event);
        setConnectionStatus('disconnected');
        setIsRecording(false);
        setIsProcessing(false);
        
        // Attempt to reconnect if not closed normally and not too many attempts
        if (event.code !== 1000 && reconnectAttemptsRef.current < 3) {
          reconnectAttemptsRef.current++;
          const delay = Math.pow(2, reconnectAttemptsRef.current) * 1000; // Exponential backoff
          console.log(`Attempting to reconnect in ${delay/1000} seconds...`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log('Attempting to reconnect...');
            connectWebSocket(patientId);
          }, delay);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
        setError('Connection error. Please try again later.');
        setIsRecording(false);
        setIsProcessing(false);
      };
      
      wsRef.current = ws;
    } catch (e) {
      console.error('Error creating WebSocket:', e);
      setConnectionStatus('error');
      setError('Failed to establish connection. Please try again later.');
    }
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
        const response = await fetch(`/api/get-conversation-audio?conversation_id=${
          reminder.message_type === 'appointment' 
            ? `appointment_${reminder.patient_name}`
            : `medication_${reminder.patient_name}`
        }`);
        
        if (response.ok) {
          const data = await response.json();
          if (data.audio_data) {
            // Save the audio data for potential playback
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
      // Create a result object for the parent component
      const result: any = {
        status: 'completed',
        type: reminder?.message_type,
        patient_name: reminder?.patient_name,
        datetime: new Date(),
        audioAvailable: !!audioData,
        audioData: audioData, // Include actual audio data in the result
        doctor_name: undefined,
        appointment_datetime: undefined
      };
      
      if (reminder?.message_type === 'appointment') {
        result.doctor_name = reminder.details.doctor_name;
        result.appointment_datetime = reminder.details.datetime;
      } else if (reminder?.message_type === 'medication') {
        result.medication_name = reminder.details.medication_name;
        result.dosage = reminder.details.dosage;
        result.adherence = "Confirmed";
      }
      
      onComplete(result);
    }
  };

  const playAudio = async (base64Audio: string, sampleRate: number = 16000, encoding: 'PCM_FLOAT' | 'PCM_16' = 'PCM_FLOAT') => {
    try {
      if (!base64Audio) {
        console.error("No audio data provided");
        return;
      }
      
      setIsSpeaking(true);
      
      // Use the improved playAudioFromBase64 function with proper error handling
      await playAudioFromBase64(base64Audio, sampleRate, encoding)
        .catch(error => {
          console.error("Error during audio playback:", error);
        })
        .finally(() => {
          setIsSpeaking(false);
        });
    } catch (error) {
      console.error("Error playing audio:", error);
      setIsSpeaking(false);
    }
  };

  const replayAudio = async (message: Message) => {
    if (message.audio && message.sampleRate) {
      // Audio data is included in the message object
      await playAudio(message.audio, message.sampleRate, 'PCM_FLOAT');
    } else if (audioData && conversationId) {
      // Fall back to fetching from the conversation's stored audio
      try {
        const response = await fetch(`/api/audio-data/${conversationId}`);
        if (response.ok) {
          const data = await response.json();
          if (data && data.audio) {
            await playAudio(data.audio, data.sample_rate || 16000, data.encoding || 'PCM_FLOAT');
          }
        }
      } catch (error) {
        console.error("Error fetching audio data:", error);
      }
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

  return (
    <Card className={`w-full ${className}`}>
      <CardBody>
        <div className="flex flex-col h-[600px]">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-lg font-medium">
              {reminder?.message_type === 'appointment' ? 'Appointment' : 'Medication'} Conversation
            </h3>
            {getConnectionStatusIndicator()}
          </div>
          
          {/* Messages area */}
          <div className="flex-1 overflow-y-auto mb-4 space-y-3 p-2 bg-gray-50 dark:bg-gray-800 rounded-lg">
            {messages.map((message, index) => (
              <div 
                key={index} 
                className={`p-3 rounded-lg ${
                  message.role === 'user' 
                    ? 'bg-blue-100 dark:bg-blue-900 ml-auto max-w-[75%] text-blue-900 dark:text-blue-100' 
                    : message.role === 'system'
                    ? 'bg-yellow-50 dark:bg-yellow-900/50 border border-yellow-200 dark:border-yellow-800 max-w-[85%] text-yellow-900 dark:text-yellow-100' 
                    : 'bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 shadow-sm max-w-[85%] text-gray-800 dark:text-gray-100'
                }`}
              >
                <div className="flex justify-between items-start mb-1">
                  <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                    {message.role === 'user' ? 'You' : message.role === 'system' ? 'System' : 'Assistant'}
                  </span>
                  <span className="text-xs text-gray-400 dark:text-gray-500">
                    {message.timestamp.toLocaleTimeString()}
                  </span>
                </div>
                
                <p className="whitespace-pre-line">{message.content}</p>
                
                {(message.audio || audioData) && message.role === 'agent' && (
                  <button 
                    onClick={() => replayAudio(message)}
                    className="mt-2 text-xs flex items-center text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 disabled:opacity-50"
                    disabled={isSpeaking}
                  >
                    <SpeakerWaveIcon className="h-3 w-3 mr-1" />
                    {isSpeaking ? 'Playing...' : 'Play audio'}
                  </button>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
            
            {/* Processing indicator */}
            {isProcessing && !isRecording && (
              <div className="flex items-center space-x-2 p-3 bg-gray-100 dark:bg-gray-700 rounded-lg animate-pulse">
                <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                <span className="text-sm text-gray-500 dark:text-gray-400">Processing...</span>
              </div>
            )}
            
            {/* Speaking indicator */}
            {isSpeaking && (
              <div className="flex items-center space-x-2 p-3 bg-green-50 dark:bg-green-900/30 rounded-lg">
                <div className="w-2 h-2 bg-green-600 rounded-full animate-pulse"></div>
                <div className="w-2 h-2 bg-green-600 rounded-full animate-pulse delay-100"></div>
                <div className="w-2 h-2 bg-green-600 rounded-full animate-pulse delay-200"></div>
                <span className="text-sm text-gray-600 dark:text-gray-300">Speaking...</span>
              </div>
            )}
          </div>

          {/* Available slots section */}
          {availableSlots.length > 0 && (
            <div className="mb-4">
              <h3 className="text-md font-semibold mb-2">Available Slots:</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-40 overflow-y-auto bg-white dark:bg-gray-800 p-2 rounded-lg border border-gray-200 dark:border-gray-700">
                {availableSlots.map((doctor, index) => (
                  <Card key={index} className="bg-gray-50 dark:bg-gray-700">
                    <CardBody className="p-2">
                      <h4 className="font-semibold">{doctor.doctor_name}</h4>
                      <p className="text-sm">{doctor.specialty}</p>
                      {doctor.available_slots?.length > 0 ? (
                        <div className="max-h-24 overflow-y-auto mt-1">
                          <p className="text-xs text-gray-500 dark:text-gray-400">Available slots:</p>
                          <div className="grid grid-cols-2 gap-1 mt-1">
                            {doctor.available_slots.slice(0, 4).map((slot: string, i: number) => (
                              <div key={i} className="text-xs p-1 bg-white dark:bg-gray-600 rounded border border-gray-200 dark:border-gray-500">
                                {new Date(slot).toLocaleString(undefined, {
                                  weekday: 'short',
                                  day: 'numeric',
                                  month: 'short',
                                  hour: '2-digit',
                                  minute: '2-digit'
                                })}
                              </div>
                            ))}
                          </div>
                          {doctor.available_slots.length > 4 && (
                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                              +{doctor.available_slots.length - 4} more slots
                            </p>
                          )}
                        </div>
                      ) : (
                        <p className="text-xs text-gray-500 dark:text-gray-400">No available slots</p>
                      )}
                    </CardBody>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {/* Action buttons */}
          <div className="flex items-center space-x-2 mt-2">
            <Button
              color={isRecording ? "danger" : "primary"}
              onClick={toggleRecording}
              className="flex-1 justify-center"
              isLoading={isProcessing && !isRecording}
              isDisabled={(isProcessing && !isRecording) || connectionStatus !== 'connected' || isSpeaking}
              startContent={isRecording ? <StopIcon className="h-5 w-5" /> : <MicrophoneIcon className="h-5 w-5" />}
            >
              {isRecording ? "Stop Recording" : (isProcessing ? 'Processing...' : 'Start Speaking')}
            </Button>
            <Button
              color="secondary"
              onClick={finishConversation}
              isDisabled={isRecording || messages.length < 2 || connectionStatus !== 'connected' || isSpeaking}
              startContent={<CheckCircleIcon className="h-5 w-5" />}
            >
              Complete
            </Button>
          </div>
          
          {/* Error messages */}
          {error && (
            <div className="mt-2 p-2 bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-300 rounded-md text-sm border border-red-200 dark:border-red-800">
              {error}
            </div>
          )}
          
          {/* Status messages */}
          <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
            {isRecording && (
              <div className="flex items-center">
                <span className="inline-block h-2 w-2 mr-1 bg-red-600 rounded-full animate-pulse"></span>
                Recording... Speak clearly into your microphone.
              </div>
            )}
            {connectionStatus === 'connected' && !isRecording && !isProcessing && !isSpeaking && (
              <p>Ready for conversation. Click "Start Speaking" to begin.</p>
            )}
          </div>
        </div>
      </CardBody>
    </Card>
  );
}