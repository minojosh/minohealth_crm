"use client";
import { useState, useEffect, useRef } from 'react';
import { Button } from '@heroui/button';
import { Card, CardBody } from '@heroui/card';
import { ReminderResponse } from '../../app/appointment-manager/types';
import { MicrophoneIcon, StopIcon, SpeakerWaveIcon, CheckCircleIcon } from '@heroicons/react/24/solid';
import { appointmentManagerApi } from '../../app/appointment-manager/api';
import { playAudioFromBase64 } from '../../app/api/audio';
import { motion, AnimatePresence } from 'framer-motion';

interface Message {
  role: 'user' | 'system' | 'agent';
  content: string;
  timestamp: Date;
  audio?: string; // Base64 audio data for playback
  sampleRate?: number;
}

interface MedicationConversationProps {
  reminder?: ReminderResponse;
  onComplete?: (result: any) => void;
  className?: string;
}

export function MedicationConversation({ reminder, onComplete, className = "" }: MedicationConversationProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('connecting');
  const [error, setError] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [audioData, setAudioData] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const latestAudioRef = useRef<string | null>(null);
  const speechSynthesisRef = useRef<SpeechSynthesisUtterance | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  // Connect to the conversation when a reminder is selected
  useEffect(() => {
    if (reminder) {
      // Create a unique ID for this conversation based on reminder details
      const id = `${reminder.message_type}_${reminder.details.patient_id || 'unknown'}_${Date.now()}`;
      setConversationId(id);
      connectWebSocket(reminder.details.patient_id?.toString() || '1');
      
      // Fetch initial greeting from API instead of using hardcoded text
      fetchInitialGreeting(reminder);
    }
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      stopMediaTracks();
    };
  }, [reminder]);

  // Stop media tracks when component unmounts or when recording stops
  const stopMediaTracks = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

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

  // Fetch initial greeting from the API
  const fetchInitialGreeting = async (reminderData: ReminderResponse) => {
    try {
      setIsProcessing(true);
      // Fetch the initial greeting from the API
      const response = await fetch(`/api/audio-data/initial-greeting?type=${reminderData.message_type}&patientId=${reminderData.details.patient_id || '1'}`);
      
      if (response.ok) {
        const data = await response.json();
        if (data.greeting) {
          // Add the greeting message
          const welcomeMessage = {
            role: 'system' as const,
            content: data.greeting,
            timestamp: new Date(),
            audio: data.audioData || null
          };
          
          setMessages([welcomeMessage]);
          
          // Play the audio if available
          if (data.audioData) {
            setAudioData(data.audioData);
            playGreetingAudio(data.audioData);
          }
        } else {
          // Fallback to default greeting if API doesn't return one
          const defaultGreeting = reminderData.details.medication_name 
            ? `Hello ${reminderData.patient_name}! I'm your medication assistant. This is a reminder about your medication: ${reminderData.details.medication_name}, dosage: ${reminderData.details.dosage || 'prescribed dosage'}, frequency: ${reminderData.details.frequency || 'as prescribed'}.`
            : `Hello ${reminderData.patient_name}! I'm your medication assistant. This is a reminder about your medication.`;
          
          setMessages([{
            role: 'system',
            content: defaultGreeting,
            timestamp: new Date()
          }]);
        }
      } else {
        throw new Error('Failed to fetch greeting');
      }
    } catch (error) {
      console.error("Error fetching initial greeting:", error);
      // Fallback to default greeting
      if (reminderData) {
        const defaultGreeting = `Hello ${reminderData.patient_name}! I'm your medication assistant. This is a reminder about your medication.`;
        
        setMessages([{
          role: 'system',
          content: defaultGreeting,
          timestamp: new Date()
        }]);
      }
    } finally {
      setIsProcessing(false);
    }
  };

  // Play greeting audio with visual feedback
  const playGreetingAudio = async (audioBase64: string) => {
    try {
      setIsSpeaking(true);
      await playAudioFromBase64(audioBase64, undefined, (utterance) => {
        speechSynthesisRef.current = utterance;
      });
    } catch (error) {
      console.error("Error playing audio:", error);
    } finally {
      setIsSpeaking(false);
      speechSynthesisRef.current = null;
    }
  };

  const connectWebSocket = (patientId: string) => {
    // WebSocket connection logic similar to AppointmentConversation
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
    // Add auth token to prevent 403 Forbidden errors
    const wsEndpoint = `${wsUrl}/ws/conversation?patient_id=${patientId}&type=medication&token=audio_access_token_2023`;
    
    try {
      wsRef.current = new WebSocket(wsEndpoint);
      
      wsRef.current.onopen = () => {
        console.log("WebSocket connection established");
        setConnectionStatus('connected');
        reconnectAttemptsRef.current = 0;
        setError(null);
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log("WebSocket message received:", data);
          
          if (data.type === 'audio' && data.audio) {
            // Handle audio response
            setAudioData(data.audio);
            handleAudioResponse(data.audio, data.text || 'Audio response', data.sample_rate);
          } else if (data.type === 'message' && data.text) {
            // Handle text message
            addAgentMessage(data.text, data.audio);
          } else if (data.type === 'transcription' && data.text) {
            // Handle transcription of user's speech
            addUserMessage(data.text);
          } else if (data.type === 'error') {
            setError(data.message || 'An error occurred');
          }
        } catch (err) {
          console.error("Error parsing WebSocket message:", err);
        }
      };
      
      wsRef.current.onclose = (event) => {
        console.log("WebSocket connection closed", event);
        setConnectionStatus('disconnected');
        // Implement reconnection logic
        if (reconnectAttemptsRef.current < 5) {
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current += 1;
            connectWebSocket(patientId);
          }, 2000 * Math.pow(2, reconnectAttemptsRef.current));
        }
      };
      
      wsRef.current.onerror = (event) => {
        console.error("WebSocket error:", event);
        setConnectionStatus('error');
        setError('WebSocket connection error');
      };
      
    } catch (err) {
      console.error("Error connecting to WebSocket:", err);
      setConnectionStatus('error');
      setError('Failed to connect to conversation service');
    }
  };

  const handleAudioResponse = async (audioBase64: string, text: string, sampleRate?: number) => {
    try {
      // Add the message first
      addAgentMessage(text, audioBase64, sampleRate);
      
      // Play the audio
      setIsSpeaking(true);
      await playAudioFromBase64(audioBase64, sampleRate, (utterance) => {
        speechSynthesisRef.current = utterance;
      });
    } catch (error) {
      console.error("Error handling audio response:", error);
    } finally {
      setIsSpeaking(false);
      speechSynthesisRef.current = null;
    }
  };

  const addAgentMessage = (content: string, audio?: string, sampleRate?: number) => {
    const newMessage: Message = {
      role: 'agent',
      content,
      timestamp: new Date(),
      audio,
      sampleRate
    };
    
    setMessages(prev => [...prev, newMessage]);
  };

  const addUserMessage = (content: string) => {
    const newMessage: Message = {
      role: 'user',
      content,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, newMessage]);
  };

  const startRecording = async () => {
    if (isRecording) {
      stopRecording();
      return;
    }

    if (isSpeaking) {
      // Cancel current speaking if user wants to start speaking
      stopSpeaking();
    }
    
    try {
      setIsRecording(true);
      setError(null);
      audioChunksRef.current = [];
      
      console.log("Requesting microphone access...");
      // Get user media for recording
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      console.log("Creating media recorder...");
      // Create media recorder
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        console.log("Media recorder stopped, processing audio...");
        setIsRecording(false);
        setIsProcessing(true);
        
        try {
          if (audioChunksRef.current.length === 0) {
            console.log("No audio data recorded");
            setIsProcessing(false);
            return;
          }

          // Create audio blob and convert to base64
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
          const reader = new FileReader();
          
          reader.onloadend = async () => {
            const base64Audio = reader.result as string;
            const base64Data = base64Audio.split(',')[1]; // Remove data URL prefix
            
            console.log("Audio converted to base64, sending to server...");
            // Send audio to server for processing
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({
                type: 'audio_input',
                audio: base64Data,
                conversation_id: conversationId
              }));
            } else {
              console.warn("WebSocket not connected, using REST API fallback");
              // Fallback to REST API if WebSocket is not available
              const response = await fetch('/api/transcribe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ audio: base64Data })
              });
              
              if (response.ok) {
                const data = await response.json();
                if (data.transcription) {
                  addUserMessage(data.transcription);
                } else {
                  console.warn("No transcription received from API");
                  setError("No speech detected");
                }
              } else {
                throw new Error('Transcription failed');
              }
            }
          };
          
          reader.readAsDataURL(audioBlob);
        } catch (error) {
          console.error("Error processing recording:", error);
          setError('Failed to process your recording');
        } finally {
          setIsProcessing(false);
          stopMediaTracks();
        }
      };
      
      // Start recording with small timeslice to get data frequently
      mediaRecorder.start(1000);
      
      // Safety timeout to stop recording after max duration
      setTimeout(() => {
        if (mediaRecorderRef.current?.state === 'recording') {
          stopRecording();
        }
      }, 15000); // Maximum 15 seconds recording
      
    } catch (error) {
      console.error("Error starting recording:", error);
      setIsRecording(false);
      setError('Could not access microphone. Please ensure microphone permissions are granted.');
    }
  };

  const stopRecording = () => {
    console.log("Stopping recording manually");
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  const stopSpeaking = () => {
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    if (speechSynthesisRef.current) {
      speechSynthesisRef.current = null;
    }
    setIsSpeaking(false);
  };

  const finishConversation = async () => {
    try {
      if (isRecording) {
        stopRecording();
      }
      
      if (isSpeaking) {
        stopSpeaking();
      }
      
      // Send finish command to server
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'command',
          command: 'finish'
        }));
      }
      
      // Call onComplete with results
      if (onComplete) {
        onComplete({
          status: 'completed',
          type: 'medication',
          patient_name: reminder?.patient_name,
          messages: messages,
          audioData: latestAudioRef.current
        });
      }
    } catch (err) {
      console.error("Error finishing conversation:", err);
      setError('Error completing conversation');
    }
  };

  // Determine if user input should be disabled
  const isInputDisabled = isProcessing || connectionStatus === 'error';

  return (
    <Card className={`w-full ${className}`}>
      <CardBody className="p-4">
        <div className="flex flex-col h-[500px]">
          {/* Status indicators with Framer Motion */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <motion.div 
                initial={{ scale: 0.8 }}
                animate={{ 
                  scale: connectionStatus === 'connected' ? [1, 1.2, 1] : 1,
                  opacity: connectionStatus === 'connected' ? 1 : 0.5
                }}
                transition={{ 
                  repeat: connectionStatus === 'connecting' ? Infinity : 0,
                  duration: 1.5 
                }}
                className={`w-3 h-3 rounded-full ${connectionStatus === 'connected' ? 'bg-green-500' : 
                  connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'}`}
              />
              <span className="text-sm">
                {connectionStatus === 'connected' ? 'Connected' : 
                 connectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected'}
              </span>
            </div>
            
            {/* Audio playback indicator with stop button */}
            <AnimatePresence>
              {isSpeaking && (
                <motion.div 
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="flex items-center gap-2 text-primary"
                >
                  <SpeakerWaveIcon className="w-5 h-5" />
                  <motion.span 
                    initial={{ width: 0 }}
                    animate={{ width: 'auto' }}
                    className="text-sm whitespace-nowrap"
                  >
                    Speaking...
                  </motion.span>
                  <Button 
                    size="sm" 
                    color="danger" 
                    variant="flat" 
                    isIconOnly
                    onClick={stopSpeaking}
                    className="ml-2"
                  >
                    <StopIcon className="w-4 h-4" />
                  </Button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          
          {/* Messages area */}
          <div className="flex-grow overflow-y-auto mb-4 space-y-4 p-2">
            <AnimatePresence>
              {messages.map((message, index) => (
                <motion.div
                  key={`${message.role}-${index}`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div 
                    className={`max-w-[80%] rounded-lg p-3 ${message.role === 'user' 
                      ? 'bg-primary text-white' 
                      : message.role === 'system' 
                        ? 'bg-default-100' 
                        : 'bg-default-200'}`}
                  >
                    <p>{message.content}</p>
                    {message.audio && (
                      <Button 
                        size="sm" 
                        variant="flat" 
                        className="mt-2 gap-1"
                        onClick={() => playAudioFromBase64(message.audio || '', message.sampleRate)}
                        isDisabled={isSpeaking}
                      >
                        <SpeakerWaveIcon className="w-4 h-4" />
                        <span>Play</span>
                      </Button>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>
          
          {/* Error message */}
          <AnimatePresence>
            {error && (
              <motion.div 
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="bg-danger-100 text-danger p-2 rounded-lg mb-4"
              >
                {error}
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Recording indicator while processing */}
          <AnimatePresence>
            {isProcessing && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex justify-center items-center gap-2 text-primary mb-4"
              >
                <div className="flex gap-1">
                  <motion.div 
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ repeat: Infinity, duration: 1 }}
                    className="w-2 h-2 rounded-full bg-primary"
                  />
                  <motion.div 
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ repeat: Infinity, duration: 1, delay: 0.2 }}
                    className="w-2 h-2 rounded-full bg-primary"
                  />
                  <motion.div 
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ repeat: Infinity, duration: 1, delay: 0.4 }}
                    className="w-2 h-2 rounded-full bg-primary"
                  />
                </div>
                <span className="text-sm">Processing...</span>
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Recording controls */}
          <div className="flex justify-center gap-4 mt-auto">
            <Button
              color={isRecording ? "danger" : "primary"}
              variant={isRecording ? "solid" : "solid"}
              startContent={isRecording ? <StopIcon className="w-5 h-5" /> : <MicrophoneIcon className="w-5 h-5" />}
              isLoading={isProcessing}
              isDisabled={isInputDisabled}
              onClick={startRecording}
              className="relative"
            >
              {isRecording ? "Stop Recording" : "Start Speaking"}
              {isRecording && (
                <motion.div
                  className="absolute inset-0 rounded-lg border-2 border-danger"
                  initial={{ opacity: 0.5, scale: 1 }}
                  animate={{ opacity: 0, scale: 1.2 }}
                  transition={{ repeat: Infinity, duration: 1.5 }}
                />
              )}
            </Button>
            
            <Button
              color="secondary"
              onClick={finishConversation}
              isDisabled={isRecording || isProcessing}
              startContent={<CheckCircleIcon className="w-5 h-5" />}
            >
              Complete
            </Button>
          </div>
        </div>
      </CardBody>
    </Card>
  );
}