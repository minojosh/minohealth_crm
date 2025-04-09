import { useState, useEffect, useRef } from 'react';
import { Button } from '@heroui/button';
import { Card, CardBody } from '@heroui/card';
import { ReminderResponse } from '../../app/appointment-manager/types';
import { MicrophoneIcon, StopIcon, CheckCircleIcon, ArrowPathIcon  } from '@heroicons/react/24/solid';

interface Message {
  role: 'user' | 'system' | 'agent';
  content: string;
  timestamp: Date;
}

interface ConversationResult {
  status: string;
  [key: string]: any;
}

interface ConversationPanelProps {
  reminder?: ReminderResponse;
  onComplete?: (result: ConversationResult) => void;
  className?: string;
}

interface MedicationConversationProps extends ConversationPanelProps {
}

export function MedicationConversation({ reminder, onComplete, className = "" }: MedicationConversationProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [textInput, setTextInput] = useState<string>('');

  // WebSocket and Audio refs
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const MAX_RETRIES = 3;
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Message handling
  const addMessage = (role: 'user' | 'agent', content: string) => {
    setMessages(prev => [...prev, {
      role,
      content,
      timestamp: new Date()
    }]);
  };

  // WebSocket setup
  useEffect(() => {
    let isComponentMounted = true;

    const connectWebSocket = () => {
      if (!isComponentMounted || reconnectAttemptsRef.current >= MAX_RETRIES) {
        setError('Connection failed. Please refresh the page.');
        return;
      }

      // Use a consistent ID format for the WebSocket connection
      // If reminder is undefined, use 'default' as the ID
      const reminderId = reminder?.details?.id || 'default';
      const wsUrl = `${process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'}/ws/${reminderId}`;
      
      console.log(`Connecting to WebSocket at: ${wsUrl}`);
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        if (isComponentMounted) {
          setWsConnected(true);
          reconnectAttemptsRef.current = 0;
          setError(null);
          if (reminder) {
            console.log('Sending reminder context:', reminder);
            wsRef.current?.send(JSON.stringify({
              type: 'context',
              context: JSON.stringify(reminder)
            }));
          }
        }
      };

      wsRef.current.onmessage = async (event) => {
        const data = JSON.parse(event.data);
        console.log('Received WebSocket message:', data);

        switch (data.type) {
          case 'message':
            addMessage(data.role || 'agent', data.text);
            if (data.role !== 'system') {
              setIsProcessing(false);
            }
            break;
          case 'summary':
            if (data.text === "null") {
              addMessage('agent', "No medication update was recorded.");
            } else {
              try {
                const summary = JSON.parse(data.text);
                addMessage('agent', `Medication status updated: ${summary.status}`);
              } catch (e) {
                console.error('Error parsing summary:', e);
              }
            }
            if (onComplete) {
              onComplete({
                status: 'completed',
                type: 'medication',
                patient_name: reminder?.patient_name,
                messages: messages
              });
            }
            break;
          case 'error':
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
  }, [reminder]);

  const startRecording = async () => {
    try {
      // First make sure any existing recording is stopped
      if (isRecording) {
        stopRecording();
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

  const stopRecording = () => {
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

  const finishConversation = () => {
    if (isRecording) {
      stopRecording();
    }
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'end_conversation'
      }));
      setIsProcessing(true);
    } else {
      setError('Connection lost. Please refresh the page.');
    }
  };

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

  return (
    <Card className={`w-full ${className}`}>
      <CardBody className="p-4">
        <div className="flex flex-col h-[500px]">
          {/* Messages container */}
          <div className="flex-1 overflow-y-auto mb-4 space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] p-3 rounded-lg ${
                    message.role === 'user'
                      ? 'bg-primary text-white'
                      : 'bg-gray-100 dark:bg-gray-800'
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
                disabled={isProcessing || !wsConnected}
              />
              <Button
                color="primary"
                onClick={sendTextMessage}
                disabled={isProcessing || !wsConnected || !textInput.trim()}
              >
                Send
              </Button>
            </div>
            
            <div className="flex items-center justify-between">
              <Button
                color={isRecording ? 'danger' : 'primary'}
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isProcessing || !wsConnected}
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
            <div className="text-danger text-sm mt-2">{error}</div>
          )}

          {/* Processing indicator */}
          {isProcessing && (
                        <span className="text-sm text-blue-500 flex items-center">
                          <ArrowPathIcon className="w-4 h-4 mr-1 animate-spin" />
                          Processing...
                        </span>
                      )}

          {/* Connection status */}
          {!wsConnected && (
            <div className="text-warning text-sm mt-2">Connecting...</div>
          )}
        </div>
      </CardBody>
    </Card>
  );
}