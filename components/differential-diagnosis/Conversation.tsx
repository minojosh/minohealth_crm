"use client";

import React, { useState, useEffect, useRef } from 'react';
import { ArrowLeftIcon, MicrophoneIcon, StopIcon, PaperAirplaneIcon } from '@heroicons/react/24/solid';
import { Card, CardBody } from "@heroui/card";
import { Button } from "@heroui/button";
import { Textarea } from "@heroui/input";
import DiagnosisSummary from '@/components/differential-diagnosis/DiagnosisSummary';

// Config values that can be adjusted
const WEBSOCKET_URL = process.env.NEXT_PUBLIC_WEBSOCKET_URL || 'ws://localhost:8000';
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const AUDIO_CHUNK_SIZE = 30000; // 30 seconds of audio
const MAX_RETRIES = 3;

// Audio context for TTS streaming
let audioContext: AudioContext | null = null;
let audioQueue: AudioBuffer[] = [];
let isPlaying = false;

interface Message {
  role: 'user' | 'agent';
  text: string;
  type?: string;
  isPartial?: boolean;
}

interface DiagnosisResult {
  primaryDiagnosis: string;
  differentialDiagnoses: string[];
  recommendedTests: string[];
  fullSummary: string;
  patientId: number;
  conditionId?: number;
}

interface DiagnosisConversationProps {
  patientId: number;
  onBack: () => void;
}

const DiagnosisConversation = ({ patientId, onBack }: DiagnosisConversationProps) => {
  // State management
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [audioChunks, setAudioChunks] = useState<Blob[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);
  const [wsRetries, setWsRetries] = useState(0);
  const [inputText, setInputText] = useState('');
  const [diagnosisSummary, setDiagnosisSummary] = useState<DiagnosisResult | null>(null);
  const [isFetchingSummary, setIsFetchingSummary] = useState(false);
  const [ttsStream, setTtsStream] = useState<ReadableStreamDefaultReader | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const recordingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const audioBuffersRef = useRef<AudioBuffer[]>([]);
  const currentlyPlayingRef = useRef<boolean>(false);

  // Initialize audio context when needed
  const getAudioContext = () => {
    if (!audioContext) {
      audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    return audioContext;
  };

  // Connect to WebSocket when component mounts
  useEffect(() => {
    connectWebSocket();
    
    // Clean up when component unmounts
    return () => {
      if (websocket) {
        websocket.close();
      }
      if (mediaRecorder) {
        handleStopRecording();
      }
      if (recordingTimeoutRef.current) {
        clearTimeout(recordingTimeoutRef.current);
      }
      if (ttsStream) {
        ttsStream.cancel();
      }
      // Stop any audio playback
      if (audioContext) {
        audioBuffersRef.current = [];
        currentlyPlayingRef.current = false;
        audioContext.close();
        audioContext = null;
      }
    };
  }, [patientId]);

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Add a keepalive function to ensure the WebSocket stays connected
  useEffect(() => {
    const pingInterval = setInterval(() => {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        // Send a ping message to keep the connection alive
        try {
          websocket.send(JSON.stringify({ type: 'ping' }));
        } catch (err) {
          console.error('Error sending ping:', err);
        }
      }
    }, 30000); // Send ping every 30 seconds

    return () => {
      clearInterval(pingInterval);
    };
  }, [websocket]);

  // // Function to play TTS audio buffers sequentially
  // const playNextAudioBuffer = () => {
  //   if (!audioBuffersRef.current.length || !audioContext || currentlyPlayingRef.current) return;
    
  //   currentlyPlayingRef.current = true;
  //   const buffer = audioBuffersRef.current.shift();
  //   if (!buffer) {
  //     currentlyPlayingRef.current = false;
  //     return;
  //   }
    
  //   const source = audioContext.createBufferSource();
  //   source.buffer = buffer;
  //   source.connect(audioContext.destination);
  //   source.onended = () => {
  //     currentlyPlayingRef.current = false;
  //     playNextAudioBuffer();
  //   };
  //   source.start(0);
  // };

  // // TTS streaming function
  // const streamTextToSpeech = async (text: string) => {
  //   try {
  //     const baseUrl = API_URL.replace(/\/+$/, '');
      
  //     // First try the streaming TTS endpoint
  //     try {
  //       const response = await fetch(`${baseUrl}/tts-stream`, {
  //         method: 'POST',
  //         headers: { 'Content-Type': 'application/json' },
  //         body: JSON.stringify({ 
  //           text,
  //           language: "en" // Provide language parameter
  //         }),
  //       });

  //       if (!response.ok) {
  //         throw new Error(`TTS streaming request failed with status: ${response.status}`);
  //       }

  //       // Process the streaming response
  //       const reader = response.body?.getReader();
  //       if (!reader) {
  //         throw new Error('No reader available from response');
  //       }

  //       // Read chunks from the stream
  //       let decoder = new TextDecoder();
  //       while (true) {
  //         const { value, done } = await reader.read();
  //         if (done) break;
          
  //         // Decode the text stream
  //         const chunk = decoder.decode(value, { stream: true });
          
  //         // Process each line (each chunk might contain multiple JSON objects)
  //         const lines = chunk.split('\n').filter(line => line.trim());
  //         for (const line of lines) {
  //           try {
  //             const jsonChunk = JSON.parse(line);
              
  //             if (jsonChunk.error) {
  //               console.error('TTS stream error:', jsonChunk.error);
  //               continue;
  //             }
              
  //             if (jsonChunk.chunk) {
  //               // Decode the base64 audio
  //               const audioData = atob(jsonChunk.chunk);
  //               const audioBytes = new Uint8Array(audioData.length);
  //               for (let i = 0; i < audioData.length; i++) {
  //                 audioBytes[i] = audioData.charCodeAt(i);
  //               }
                
  //               // Create audio context and decode
  //               const ctx = getAudioContext();
  //               ctx.decodeAudioData(
  //                 audioBytes.buffer as ArrayBuffer,
  //                 (audioBuffer) => {
  //                   audioBuffersRef.current.push(audioBuffer);
  //                   if (!currentlyPlayingRef.current) {
  //                     playNextAudioBuffer();
  //                   }
  //                 },
  //                 (error) => {
  //                   console.error('Error decoding audio chunk:', error);
  //                 }
  //               );
  //             }
  //           } catch (e) {
  //             console.error('Error parsing JSON chunk:', e, line);
  //           }
  //         }
  //       }
  //     } catch (streamingError) {
  //       // Streaming failed, fall back to regular TTS
  //       console.warn('TTS streaming failed, falling back to regular TTS:', streamingError);
        
  //       // Use regular TTS endpoint as fallback
  //       const fallbackResponse = await fetch(`${baseUrl}/tts`, {
  //         method: 'POST',
  //         headers: { 'Content-Type': 'application/json' },
  //         body: JSON.stringify({ 
  //           text,
  //           speaker: "en-US" 
  //         }),
  //       });

  //       if (!fallbackResponse.ok) {
  //         throw new Error(`Fallback TTS request failed with status: ${fallbackResponse.status}`);
  //       }

  //       const result = await fallbackResponse.json();
        
  //       if (result.success && result.audio) {
  //         try {
  //           // Convert base64 audio to arraybuffer
  //           const audioData = atob(result.audio);
  //           const audioBytes = new Uint8Array(audioData.length);
  //           for (let i = 0; i < audioData.length; i++) {
  //             audioBytes[i] = audioData.charCodeAt(i);
  //           }
            
  //           // Decode audio
  //           const ctx = getAudioContext();
  //           ctx.decodeAudioData(
  //             audioBytes.buffer as ArrayBuffer,
  //             (audioBuffer) => {
  //               audioBuffersRef.current.push(audioBuffer);
  //               if (!currentlyPlayingRef.current) {
  //                 playNextAudioBuffer();
  //               }
  //             },
  //             (error) => {
  //               console.error('Error decoding audio data:', error);
  //             }
  //           );
  //         } catch (error) {
  //           console.error('Error processing audio data:', error);
  //         }
  //       } else {
  //         console.error('TTS request did not return valid audio data');
  //       }
  //     }
  //   } catch (err) {
  //     console.error('All TTS attempts failed:', err);
  //   }
  // };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Modified WebSocket connection function with better error handling
  const connectWebSocket = () => {
    // Close existing connection if any
    if (websocket) {
      try {
        websocket.close();
      } catch (err) {
        console.error('Error closing existing WebSocket:', err);
      }
    }

    try {
      // Create new connection
      const ws = new WebSocket(`${WEBSOCKET_URL}/ws/diagnosis/${patientId}`);
      
      // Set a connection timeout
      const connectionTimeout = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          setError('Connection timeout. Please refresh and try again.');
          try {
            ws.close();
          } catch (e) {
            console.error('Error closing timed out connection:', e);
          }
        }
      }, 10000);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        clearTimeout(connectionTimeout);
        setError(null);
        setWsRetries(0);
        
        // Initialize with empty messages array
        // The greeting will come from the backend
        setMessages([]);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('Received message:', data);
          
          // Handle different message types
          if (data.type === 'message' || data.type === 'partial_message') {
            const isPartial = data.type === 'partial_message';
            const messageText = data.text || '';
            
            // Update existing partial message or add new message
            setMessages(prevMessages => {
              const lastMessage = prevMessages[prevMessages.length - 1];
              
              // If we have a partial message from the agent, update it
              if (lastMessage && lastMessage.role === 'agent' && lastMessage.isPartial) {
                const updatedMessages = [...prevMessages];
                updatedMessages[updatedMessages.length - 1] = {
                  ...lastMessage,
                  text: messageText,
                  isPartial: isPartial
                };
                return updatedMessages;
              } else {
                // Add new message
                return [...prevMessages, {
                  role: data.role,
                  text: messageText,
                  type: data.type,
                  isPartial: isPartial
                }];
              }
            });
            
            // If this is a new or updated message from the agent, speak it with TTS
            if (data.role === 'agent' && messageText) {
              // For determining if we should play the full message or just the new portion
              const currentMessages = [...messages];
              const lastMessage = currentMessages.length > 0 ? currentMessages[currentMessages.length - 1] : null;

              // Only attempt TTS if we have some text to speak
             // if (messageText.trim().length > 0) {
                // For new messages, speak the whole thing
                // if (!lastMessage || lastMessage.role !== 'agent') {
                //   streamTextToSpeech(messageText);
                // } 
                // For existing messages that got updated, only speak the new part
                // else if (lastMessage && lastMessage.text && messageText.length > lastMessage.text.length) {
                //  const newTextPortion = messageText.substring(lastMessage.text.length);
                //  if (newTextPortion.trim().length > 0) {
                //    streamTextToSpeech(newTextPortion);
                //  }
                // }
              // }
            }
            
            if (data.type === 'message') {
              setIsProcessing(false);
            }
          } else if (data.type === 'error') {
            setError(data.message);
            setIsProcessing(false);
          } else if (data.type === 'transcription' || data.type === 'partial_transcription') {
            // Update the last user message with transcription
            setMessages(prevMessages => {
              const lastUserMessageIndex = Array.from(prevMessages).reverse()
                .findIndex(msg => 
                  msg.role === 'user' && (msg.type === 'audio_processing' || msg.type === 'partial_transcription')
                );
              
              if (lastUserMessageIndex >= 0) {
                const actualIndex = prevMessages.length - 1 - lastUserMessageIndex;
                const updatedMessages = [...prevMessages];
                updatedMessages[actualIndex] = {
                  role: 'user',
                  text: data.text,
                  type: data.type,
                  isPartial: data.type === 'partial_transcription'
                };
                return updatedMessages;
              }
              
              return prevMessages;
            });
            
            if (data.type === 'transcription') {
              setIsProcessing(false);
            }
          } else if (data.type === 'diagnosis_summary' || data.type === 'partial_diagnosis_summary') {
            try {
              let summary: DiagnosisResult;
              // Check if data.text is already an object (ideal) or a string (current case)
              if (typeof data.text === 'string') {
                  // Attempt to parse the markdown string
                  summary = parseMarkdownSummary(data.text, patientId);
                  console.info("Parsed markdown summary from string.");
              } else if (typeof data.text === 'object' && data.text !== null) {
                  // If it's already an object, use it (ensure it fits DiagnosisResult structure)
                  summary = {
                      primaryDiagnosis: data.text.primaryDiagnosis || "Not specified",
                      differentialDiagnoses: data.text.differentialDiagnoses || [],
                      recommendedTests: data.text.recommendedTests || [],
                      fullSummary: data.text.fullSummary || JSON.stringify(data.text), // Fallback for full summary
                      patientId: patientId
                  };
                   console.info("Received pre-parsed summary object.");
              } else {
                  console.warn('Received diagnosis summary in unexpected format:', data.text);
                   summary = {
                      primaryDiagnosis: "Error: Invalid Format",
                      differentialDiagnoses: [],
                      recommendedTests: [],
                      fullSummary: "Could not display summary due to invalid format.",
                      patientId: patientId
                   }
              }

              // Update any "Generating summary..." message
              setMessages(prevMessages => {
                const lastIndex = prevMessages.length - 1;
                if (lastIndex >= 0 && 
                    prevMessages[lastIndex].role === 'agent' && 
                    prevMessages[lastIndex].type === 'system' &&
                    prevMessages[lastIndex].isPartial) {
                  const updatedMessages = [...prevMessages];
                  updatedMessages.pop(); // Remove the temporary message
                  return updatedMessages;
                }
                return prevMessages;
              });
              
              // Set the diagnosis summary state
              setDiagnosisSummary(summary);
              
              if (data.type === 'diagnosis_summary') {
                setIsProcessing(false);
                setIsFetchingSummary(false);
              }
            } catch (e) {
              console.error('Error handling diagnosis summary:', e);
              setError('Failed to process diagnosis summary');
              setIsProcessing(false);
              setIsFetchingSummary(false);
            }
          } else if (data.type === 'pong') {
            // Handle pong response (keepalive)
            console.log('Received pong from server');
          }
        } catch (jsonError) {
          console.error('Error parsing WebSocket message:', jsonError, event.data);
          setError('Received invalid data from server');
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('Connection error. Please try again.');
        setIsProcessing(false);
        setIsFetchingSummary(false);
        
        // Try to reconnect after a delay
        if (wsRetries < MAX_RETRIES) {
          setTimeout(() => {
            setWsRetries(prev => prev + 1);
            connectWebSocket();
          }, 2000 * (wsRetries + 1)); // Exponential backoff
        }
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
      };
      
      setWebsocket(ws);
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setError('Failed to connect. Please check your network connection and try again.');
    }
  };

  const handleStartRecording = async () => {
    try {
      // First make sure any existing recording is stopped
      if (isRecording) {
        handleStopRecording();
      }
      
      if (!websocket || websocket.readyState !== WebSocket.OPEN) {
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
              if (websocket?.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({
                  type: 'message',
                  text: result.transcription
                }));
                
                // Add message locally for immediate feedback
                setMessages(prev => [...prev, { role: 'user', text: result.transcription }]);
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

  const sendMessage = () => {
    if (!inputText.trim() || !websocket || websocket.readyState !== WebSocket.OPEN) return;
    
    // Add message to conversation
    setMessages(prevMessages => [...prevMessages, {
      role: 'user',
      text: inputText,
      type: 'message'
    }]);
    
    // Send message to server
    websocket.send(JSON.stringify({
      type: 'message',
      text: inputText
    }));
    
    setInputText('');
    setIsProcessing(true);
  };

  const sendInitialContext = (context: string) => {
    if (!websocket || websocket.readyState !== WebSocket.OPEN) return;
    
    websocket.send(JSON.stringify({
      type: 'context',
      context
    }));
    
    setIsProcessing(true);
  };

  const endConversation = () => {
    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
      // Try to reconnect first if websocket isn't open
      setError('Connection lost. Attempting to reconnect...');
      connectWebSocket();
      
      // Set a timeout to retry the operation after reconnection
      setTimeout(() => {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
          setError(null);
          endConversation(); // Retry the operation
        } else {
          setError('Failed to reconnect. Please try again later.');
        }
      }, 2000);
      
      return;
    }
    
    setIsFetchingSummary(true);
    setIsProcessing(true);

    try {
      // Get the full conversation text to send to the backend
      const conversationText = messages
        .filter(msg => msg.type !== 'audio_processing' && !msg.isPartial) // Filter out processing messages and partials
        .map(msg => `${msg.role === 'user' ? 'Patient' : 'Doctor'}: ${msg.text}`)
        .join('\n');

      // Add a final message showing that we're generating a summary
      setMessages(prev => [...prev, {
        role: 'agent',
        text: 'Generating diagnosis summary...',
        type: 'system',
        isPartial: true
      }]);

      // Add event listeners to detect disconnection
      const closeHandler = (event: CloseEvent) => {
        console.warn('WebSocket closed during summary generation:', event); // Changed from error to warn
        setError('Connection lost during summary generation. The backend might still be processing. Please wait or try ending again later.');
        setIsProcessing(false);
        setIsFetchingSummary(false);

        // Remove the handler
        if (websocket) {
          websocket.removeEventListener('close', closeHandler);
        }
      };

      // Add temporary close handler
      websocket.addEventListener('close', closeHandler);

      // Add an error handler specifically for this critical operation
      const summaryErrorHandler = (e: Event) => {
        console.error('WebSocket error during summary generation:', e);
        setError('Error generating summary. Please try again.');
        setIsProcessing(false);
        setIsFetchingSummary(false);

        // Remove the temporary error handler
        if (websocket) {
          websocket.removeEventListener('error', summaryErrorHandler);
          websocket.removeEventListener('close', closeHandler);
        }
      };

      // Add temporary error handler
      websocket.addEventListener('error', summaryErrorHandler);

      // REMOVED FALLBACK TIMEOUT LOGIC - Relying on backend for summary
      // const fallbackTimeout = setTimeout(() => { ... }, 20000);

      // Make sure the message isn't too large for WebSocket frame
      const messagePayload = {
        type: 'end_conversation',
        conversation: conversationText
      };

      // Truncate if necessary (adjust limit as needed, e.g., 16KB typical frame limit)
      const maxPayloadSize = 16000;
      const payloadString = JSON.stringify(messagePayload);
      if (payloadString.length > maxPayloadSize) {
        console.warn("Conversation text too long, truncating for WebSocket message.");
        // Estimate truncation length needed, leaving room for JSON structure
        const overheadEstimate = 100;
        const maxTextLength = maxPayloadSize - overheadEstimate - JSON.stringify({ type: 'end_conversation', conversation: '' }).length;
        messagePayload.conversation = conversationText.substring(0, maxTextLength) + "... [Truncated]";
        websocket.send(JSON.stringify(messagePayload));
      } else {
        // Send normal message
        websocket.send(payloadString);
      }

      // Set a timeout to remove handlers if summary arrives or after a reasonable wait
      const cleanupTimeout = setTimeout(() => {
        if (websocket) {
          websocket.removeEventListener('error', summaryErrorHandler);
          websocket.removeEventListener('close', closeHandler);
        }
        // Removed clearTimeout(fallbackTimeout); as fallbackTimeout is removed
      }, 30000); // 30 seconds cleanup window

      // Clear cleanup timeout if summary processing finishes earlier
      // This needs to be handled within the onmessage handler for diagnosis_summary
      // Add a reference to the cleanup timeout to clear it when summary arrives
      // Note: This requires adding a ref or state for the timeout ID

    } catch (err) {
      console.error('Error sending end conversation command:', err);
      setError('Failed to generate summary. Please try again.');
      setIsProcessing(false);
      setIsFetchingSummary(false);
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

  // Helper function to parse markdown summary into DiagnosisResult structure
  const parseMarkdownSummary = (markdownText: string, patientId: number): DiagnosisResult => {
    let primaryDiagnosis = "Not specified";
    const differentialDiagnoses: string[] = [];
    const recommendedTests: string[] = []; // Placeholder as tests aren't explicitly in the example markdown

    try {
      // Extract Primary Diagnosis
      const primaryMatch = markdownText.match(/\\*\\*1\\.\\s*Primary Diagnosis(?:\\s*\\(Most Likely\\))?:\\*\\*\\s*\\*\\*([^*]+)\\*\\*/i);
      if (primaryMatch && primaryMatch[1]) {
        primaryDiagnosis = primaryMatch[1].trim();
      } else {
         // Fallback regex if the first one fails
        const fallbackPrimaryMatch = markdownText.match(/Primary Diagnosis:[\\s\\n]*\\*\\*([^*]+)\\*\\*/i);
        if (fallbackPrimaryMatch && fallbackPrimaryMatch[1]) {
          primaryDiagnosis = fallbackPrimaryMatch[1].trim();
        }
      }


      // Extract Differential Diagnoses (handles numbered lists)
      // Look for the start of the differential section
      const differentialSectionMatch = markdownText.match(/\\*\\*2\\.\\s*Differential Diagnoses(?:\\s*\\(Ordered by Likelihood\\))?:\\*\\*/i);
      if (differentialSectionMatch) {
          const differentialText = markdownText.substring(differentialSectionMatch.index + differentialSectionMatch[0].length);
          // Match list items like "1. **Diagnosis Name**"
          const diffMatches = differentialText.matchAll(/\\d+\\.\\s*\\*\\*([^*]+)\\*\\*/gi);
          for (const match of diffMatches) {
            if (match[1]) {
              differentialDiagnoses.push(match[1].trim());
            }
          }
      }

      // Note: Recommended tests extraction would need a specific pattern in the markdown

    } catch (e) {
      console.error("Error parsing markdown summary:", e);
      // Return default structure even if parsing fails
    }

    return {
      primaryDiagnosis,
      differentialDiagnoses,
      recommendedTests,
      fullSummary: markdownText, // Always include the full original markdown
      patientId: patientId,
    };
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
              onClick={() => setDiagnosisSummary(null)}
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
          <Button
            color="primary"
            variant="ghost"
            onClick={endConversation}
            disabled={isProcessing || messages.length <= 1 || isFetchingSummary}
          >
            {isFetchingSummary ? "Generating Summary..." : "Complete Diagnosis"}
          </Button>
        </div>
        
        <div className="flex-1 overflow-auto p-4">
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
          <div className="p-3 m-3 bg-danger-100 text-danger border border-danger rounded-md">
            {error}
          </div>
        )}
        
        <div className="p-4 border-t">
          <form
            onSubmit={e => { e.preventDefault(); sendMessage(); }}
            onKeyDown={(e) => {
              // Send message on Enter press without Shift
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault(); // Prevent default newline behavior
                sendMessage();
              }
            }}
            className="flex w-full gap-3"
          >
            <Textarea
              placeholder="Type your message..."
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              disabled={isProcessing || isRecording || isFetchingSummary}
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
                  disabled={isProcessing || isFetchingSummary}
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
                disabled={!inputText.trim() || isProcessing || isRecording || isFetchingSummary}
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