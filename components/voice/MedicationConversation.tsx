import { useState, useEffect, useRef, forwardRef, useImperativeHandle, useCallback } from 'react';
import { Button } from '@heroui/button';
import { Card, CardBody } from '@heroui/card';
import { ReminderResponse } from '../../app/appointment-manager/types';
import { MicrophoneIcon, StopIcon, CheckCircleIcon, ArrowPathIcon, BellIcon, SpeakerWaveIcon } from '@heroicons/react/24/solid';
import { playAudioFromBase64, stopAllAudio } from '../../app/api/audio';

interface Message {
  role: 'user' | 'system' | 'agent';
  content: string;
  timestamp: Date;
}

interface ConversationResult {
  status: string;
  [key: string]: any;
}

export interface MedicationConversationRef {
  setShouldKeepConnection: (value: boolean) => void;
}

interface ConversationPanelProps {
  reminder?: ReminderResponse;
  onComplete?: (result: ConversationResult) => void;
  className?: string;
}

interface MedicationConversationProps extends ConversationPanelProps {
}

interface AudioQueueItem {
  audio: string;
  sampleRate: number;
  text?: string;
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

interface SilenceDetectorConfig {
  minSilenceDuration: number;  // milliseconds
  silenceThreshold: number;    // threshold for Mean Absolute Value
  maxChunkDuration: number;    // milliseconds
}

class AudioChunkProcessor {
  private buffer: Float32Array[] = [];
  private isInSilence: boolean = false;
  private lastSoundTimestamp: number = Date.now();
  private chunkStartTime: number = Date.now();

  constructor(
    private config: SilenceDetectorConfig = {
      minSilenceDuration: 400,
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

export const MedicationConversation = forwardRef<MedicationConversationRef, MedicationConversationProps>(
  function MedicationConversation({ reminder, onComplete, className = "" }, ref) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [textInput, setTextInput] = useState<string>('');
  const [ttsPlaying, setTtsPlaying] = useState(false);
  const [audioQueue, setAudioQueue] = useState<AudioQueueItem[]>([]);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const audioQueueRef = useRef<AudioQueueItem[]>([]);

  // WebSocket and Audio refs
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const MAX_RETRIES = 3;
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [shouldKeepConnection, setShouldKeepConnection] = useState(true);

  // Add queue refs after other refs
  const inputQueueRef = useRef<AudioInputQueue>({
    chunks: [],
    isProcessing: false
  });

  const outputQueueRef = useRef<AudioOutputQueue>({
    chunks: [],
    isPlaying: false
  });

  // Add audio processing refs
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const transcriptionBufferRef = useRef<string>('');

  // Add currentTranscription state
  const [currentTranscription, setCurrentTranscription] = useState<string>('');

  // Auto-scroll to the latest message
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

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
              // If this is a goodbye message, set processing to true
              if (data.is_goodbye) {
                console.log('Received goodbye message, showing processing indicator');
                setIsProcessing(true);
              } else {
                setIsProcessing(false);
              }
            }
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
            setShouldKeepConnection(false); // Signal that we're done with the conversation
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

    // Add a small delay before connecting to ensure any previous connection is fully closed
    const timeoutId = setTimeout(() => {
      connectWebSocket();
    }, 500);

    return () => {
      isComponentMounted = false;
      clearTimeout(timeoutId);
      
      if (!shouldKeepConnection && wsRef.current) {
        console.log('[WebSocket] Component unmounting, closing connection');
        wsRef.current.close();
      }
      
      // Stop any audio that might be playing
      stopAllAudio();
    };
  }, [reminder, shouldKeepConnection]);

  // Cleanup effect when component unmounts
  useEffect(() => {
    return () => {
      // Clean up when component unmounts
      setTtsPlaying(false);
      
      // Ensure any playing audio is stopped
      stopAllAudio();
    };
  }, []);

  // Expose the setShouldKeepConnection method via ref
  useImperativeHandle(ref, () => ({
    setShouldKeepConnection: (value: boolean) => {
      console.log(`[MedicationConversation] Setting shouldKeepConnection to ${value}`);
      setShouldKeepConnection(value);
      
      // If setting to false, immediately close the WebSocket
      if (value === false && wsRef.current) {
        // If the connection is open, send a cleanup message before closing
        if (wsRef.current.readyState === WebSocket.OPEN) {
          console.log('[MedicationConversation] Sending cleanup message before closing connection');
          wsRef.current.send(JSON.stringify({
            type: 'client_cleanup',
            reason: 'navigation'
          }));
          
          // Short timeout to allow the message to be sent before closing
          setTimeout(() => {
            console.log('[MedicationConversation] Closing WebSocket connection after cleanup message');
            wsRef.current?.close();
            
            // Reset TTS playing state and stop any audio
            setTtsPlaying(false);
            stopAllAudio();
            
            // Stop any active recording
            if (isRecording) {
              stopRecording();
            }
          }, 100);
        } else {
          // Connection is already closed or closing
          console.log('[MedicationConversation] Connection not open, skipping cleanup message');
          
          // Reset TTS playing state and stop any audio
          setTtsPlaying(false);
          stopAllAudio();
          
          // Stop any active recording
          if (isRecording) {
            stopRecording();
          }
        }
      }
    }
  }));

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

  const stopRecording = async () => {
    try {
        // 1. Stop recording first but keep processor alive until chunks are processed
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        // 2. Process any remaining chunks in the queue
        if (inputQueueRef.current.chunks.length > 0) {
            console.log(`Processing ${inputQueueRef.current.chunks.length} remaining chunks...`);
            processNextChunk(); // This will process all chunks recursively
        }

        // 3. Now clean up audio processing nodes
        if (processorRef.current) {
            processorRef.current.onaudioprocess = null;
            processorRef.current.disconnect();
            processorRef.current = null;
        }

        if (sourceRef.current) {
            sourceRef.current.disconnect();
            sourceRef.current = null;
        }

        if (audioContextRef.current) {
            await audioContextRef.current.close();
            audioContextRef.current = null;
        }
        
        // 4. Send final transcription if we have any
        if (transcriptionBufferRef.current && transcriptionBufferRef.current.trim()) {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({
                    type: 'message',
                    text: transcriptionBufferRef.current.trim()
                }));
                addMessage('user', transcriptionBufferRef.current.trim());
            }
        }
        
        // 5. Clear transcription buffers
        transcriptionBufferRef.current = '';
        setCurrentTranscription('');
        
        // 6. Update recording state
        setIsRecording(false);
        
    } catch (error) {
        console.error("Error cleaning up recording resources:", error);
        setIsRecording(false);
    }
};

  const finishConversation = () => {
    if (isRecording) {
      stopRecording();
    }
    
    setShouldKeepConnection(false); // Signal that we're done with the conversation
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'end_conversation'
      }));
      setIsProcessing(true);
      
      // Stop any playing audio
      stopAllAudio();
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
    if (!ttsPlaying){
    setTtsPlaying(true);
    }
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
      setTtsPlaying(false);
    } finally {
      setIsPlayingAudio(false);
      console.log('[WebSocket] TTS playback completed');
    }
  }, [audioQueue]);

  // Monitor queue and process chunks
  useEffect(() => {
    if (!isPlayingAudio && audioQueue.length > 0) {
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

  return (
    <Card className={`w-full ${className}`}>
      <CardBody className="p-4">
        <div className="mb-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold flex items-center">
              <span className="mr-2">
                <BellIcon className="h-5 w-5 text-purple-500" />
              </span>
              Medication Conversation
            </h2>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
            Patient: <span className="font-medium">{reminder?.patient_name}</span> | 
            Medication: <span className="font-medium">{reminder?.details?.medication_name}</span>
          </p>
        </div>

        {/* Messages container */}
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

        {/* Connection status */}
        {!wsConnected && (
          <div className="text-warning text-sm mt-2">Connecting...</div>
        )}
      </CardBody>
    </Card>
  );
});