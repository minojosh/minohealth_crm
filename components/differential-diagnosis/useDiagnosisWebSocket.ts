import { useState, useEffect, useCallback, useRef } from 'react';

// Config values
const WEBSOCKET_URL = process.env.NEXT_PUBLIC_WEBSOCKET_URL || 'ws://localhost:8000';
const MAX_RETRIES = 3;
const RETRY_DELAY = 2000; // Base delay in ms

export interface Message {
  role: 'user' | 'agent';
  text: string;
  type?: string;
  isPartial?: boolean;
}

export interface DiagnosisResult {
  primaryDiagnosis: string;
  differentialDiagnoses: string[];
  recommendedTests: string[];
  fullSummary: string;
  patientId: number;
  conditionId?: number;
}

export default function useDiagnosisWebSocket(patientId: number) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);
  const [wsRetries, setWsRetries] = useState(0);
  const [diagnosisSummary, setDiagnosisSummary] = useState<DiagnosisResult | null>(null);
  const [isFetchingSummary, setIsFetchingSummary] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'failed' | 'disconnected'>('disconnected');
  
  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRetries >= MAX_RETRIES) {
      setError(`Failed to connect after ${MAX_RETRIES} attempts. Please check if the server is running and refresh the page.`);
      setConnectionStatus('failed');
      return;
    }

    // Close existing connection if any
    if (websocket) {
      try {
        websocket.close();
      } catch (err) {
        console.error('Error closing existing WebSocket:', err);
      }
    }

    try {
      setConnectionStatus('connecting');
      console.log(`Connecting to WebSocket at ${WEBSOCKET_URL}/ws/diagnosis/${patientId}, attempt ${wsRetries + 1}`);
      
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
          setConnectionStatus('failed');
          
          // Try to reconnect if within retry limits
          if (wsRetries < MAX_RETRIES) {
            const nextRetry = wsRetries + 1;
            setWsRetries(nextRetry);
            setTimeout(() => connectWebSocket(), RETRY_DELAY * Math.pow(2, nextRetry - 1));
          }
        }
      }, 10000);
      
      ws.onopen = () => {
        console.log('WebSocket connected successfully');
        clearTimeout(connectionTimeout);
        setError(null);
        setConnectionStatus('connected');
        setWsRetries(0);
        
        // Don't clear messages on reconnect - the server will send an initial greeting
        // that will be handled by the message handler
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
              // Try parsing the summary data
              let summary;
              if (typeof data.text === 'string') {
                try {
                  summary = JSON.parse(data.text);
                } catch (parseError) {
                  console.warn('Could not parse diagnosis summary JSON, using text directly');
                  // Create a fallback summary
                  const textSummary = data.text || "No summary provided";
                  
                  // Try to extract key information using regex patterns
                  const primaryDiagnosisMatch = textSummary.match(/primary\s*diagnosis:?\s*([^\n\.]+)/i);
                  const differentialMatches = textSummary.match(/differential\s*diagnos[ie]s:?\s*([^\n\.]+)/i);
                  const testsMatch = textSummary.match(/recommended\s*tests:?\s*([^\n\.]+)/i);
                  
                  summary = {
                    primaryDiagnosis: primaryDiagnosisMatch ? primaryDiagnosisMatch[1].trim() : "Could not parse diagnosis",
                    differentialDiagnoses: differentialMatches ? 
                      differentialMatches[1].split(',').map((d: string) => d.trim()) : [],
                    recommendedTests: testsMatch ? 
                      testsMatch[1].split(',').map((t: string) => t.trim()) : [],
                    fullSummary: textSummary,
                    patientId: patientId
                  };
                }
              } else {
                // If it's already an object, use it directly
                summary = data.text;
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
        setError('Connection error. Please ensure the server is running and try again.');
        setIsProcessing(false);
        setIsFetchingSummary(false);
        setConnectionStatus('failed');
        
        // Try to reconnect after a delay if within retry limits
        if (wsRetries < MAX_RETRIES) {
          const nextRetry = wsRetries + 1;
          setWsRetries(nextRetry);
          const delay = RETRY_DELAY * Math.pow(2, nextRetry - 1); // Exponential backoff
          console.log(`Will retry connection in ${delay}ms (attempt ${nextRetry}/${MAX_RETRIES})`);
          
          setTimeout(() => connectWebSocket(), delay);
        } else {
          setError('Maximum connection attempts reached. Please refresh the page to try again.');
        }
      };
      
      ws.onclose = (event) => {
        console.log(`WebSocket disconnected with code: ${event.code}, reason: ${event.reason}`);
        setConnectionStatus('disconnected');
        
        // Don't automatically reconnect here - let the error handler handle reconnection
        // This prevents potential double reconnection attempts
      };
      
      setWebsocket(ws);
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setError('Failed to connect. Please check your network connection and try again.');
      setConnectionStatus('failed');
      
      // Try to reconnect after a delay if within retry limits
      if (wsRetries < MAX_RETRIES) {
        const nextRetry = wsRetries + 1;
        setWsRetries(nextRetry);
        setTimeout(() => connectWebSocket(), RETRY_DELAY * Math.pow(2, nextRetry - 1));
      }
    }
  }, [patientId, websocket, wsRetries]);

  // Setup WebSocket connection on component mount
  useEffect(() => {
    // Initialize with empty messages array on first mount
    setMessages([]);
    
    connectWebSocket();
    
    // Setup ping interval to keep connection alive
    const pingInterval = setInterval(() => {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        try {
          websocket.send(JSON.stringify({ type: 'ping' }));
        } catch (err) {
          console.error('Error sending ping:', err);
        }
      }
    }, 30000);
    
    // Clean up on unmount
    return () => {
      clearInterval(pingInterval);
      if (websocket) {
        websocket.close();
      }
    };
  }, [connectWebSocket, websocket]);

  // Function to send a message via WebSocket
  const sendMessage = useCallback((text: string) => {
    if (!text.trim() || !websocket || websocket.readyState !== WebSocket.OPEN) return;
    
    // Add message to conversation
    setMessages(prevMessages => [...prevMessages, {
      role: 'user',
      text,
      type: 'message'
    }]);
    
    // Send message to server
    websocket.send(JSON.stringify({
      type: 'message',
      text
    }));
    
    setIsProcessing(true);
  }, [websocket]);

  // Function to send initial context
  const sendInitialContext = useCallback((context: string) => {
    if (!websocket || websocket.readyState !== WebSocket.OPEN) return;
    
    websocket.send(JSON.stringify({
      type: 'context',
      context
    }));
    
    setIsProcessing(true);
  }, [websocket]);

  // Function to end conversation and get diagnosis summary
  const endConversation = useCallback(() => {
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

      // Create data for the backend
      const diagnosisData = {
        conversation: conversationText,
        patientId: patientId
      };
      
      // Store the conversation locally as backup
      localStorage.setItem(`diagnosis_conversation_${patientId}`, JSON.stringify(diagnosisData));

      // Add event listeners to detect disconnection
      const closeHandler = (event: CloseEvent) => {
        console.error('WebSocket closed during summary generation:', event);
        setError('Connection lost during summary generation. Attempting to reconnect...');
        
        // Try to reconnect
        setTimeout(() => {
          connectWebSocket();
        }, 1000);
        
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
      
      // Set a timeout to clean up if no response
      const fallbackTimeout = setTimeout(() => {
        if (isFetchingSummary) {
          console.warn('Summary generation timed out');
          setError('The server did not respond. Please try again.');
          setIsProcessing(false);
          setIsFetchingSummary(false);
          
          // Clean up
          if (websocket) {
            websocket.removeEventListener('error', summaryErrorHandler);
            websocket.removeEventListener('close', closeHandler);
          }
        }
      }, 20000); // 20 second timeout
      
      // Make sure the message isn't too large for WebSocket frame
      if (conversationText.length > 50000) {
        // If message is too large, truncate it
        const truncatedText = conversationText.substring(0, 49900) + "...";
        websocket.send(JSON.stringify({
          type: 'end_conversation',
          conversation: truncatedText
        }));
      } else {
        // Send normal message
        websocket.send(JSON.stringify({
          type: 'end_conversation',
          conversation: conversationText
        }));
      }
      
      // Set a timeout to remove all handlers after a while
      setTimeout(() => {
        if (websocket) {
          websocket.removeEventListener('error', summaryErrorHandler);
          websocket.removeEventListener('close', closeHandler);
        }
        clearTimeout(fallbackTimeout);
      }, 25000); // 25 seconds
    } catch (err) {
      console.error('Error sending end conversation command:', err);
      setError('Failed to generate summary. Please try again.');
      setIsProcessing(false);
      setIsFetchingSummary(false);
    }
  }, [connectWebSocket, messages, patientId, websocket, isFetchingSummary]);

  // Return hook values and functions
  return {
    messages,
    error,
    isProcessing,
    diagnosisSummary,
    isFetchingSummary,
    connectionStatus,
    setError,
    setMessages,
    setIsProcessing,
    setDiagnosisSummary,
    sendMessage,
    sendInitialContext,
    endConversation,
    setIsFetchingSummary,
    reconnect: () => {
      setWsRetries(0);
      connectWebSocket();
    }
  };
} 