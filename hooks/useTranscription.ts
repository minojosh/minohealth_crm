import { useState, useRef, useCallback, useEffect } from 'react';
import { TranscriptionStatus } from '../app/api/types';
import { audioService } from '../services/audio/AudioService';

// Constants for error detection
const SUSPICIOUS_PATTERNS = [
  "thanks for watching", 
  "I'm sorry", 
  "thank you for watching",
  "processing your",
  "chunk",
  "please wait",
  "thank you",
  "thank you for using",
  "I am sorry",
  "I do not have",
  "Error during transcription",
  "SpeechRecognitionClient",
  "context_prompt"
];

interface TranscriptionOptions {
  apiUrl?: string;
  maxRetries?: number;
  sessionId?: string;
  useHighQualityProcessing?: boolean;
  timeoutMs?: number;
}

interface TranscriptionResult {
  startTranscription: (audioData?: string) => Promise<void>;
  cancelTranscription: () => void;
  retryTranscription: () => Promise<void>;
  transcription: string;
  isProcessing: boolean;
  error: string | null;
  status: TranscriptionStatus;
  sessionId: string | null;
}

export const useTranscription = (options: TranscriptionOptions = {}): TranscriptionResult => {
  const {
    apiUrl = process.env.NEXT_PUBLIC_API_URL || '/api',
    maxRetries = 3,
    timeoutMs = 12000,
    useHighQualityProcessing = false
  } = options;
  
  const [transcription, setTranscription] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<TranscriptionStatus>({
    isRecording: false,
    duration: 0,
    status: 'idle',
  });
  
  // Refs for tracking state and data
  const retryCountRef = useRef(0);
  const sessionIdRef = useRef<string | null>(options.sessionId || null);
  const lastAudioDataRef = useRef<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const processingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isCancelledRef = useRef<boolean>(false);
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (processingTimeoutRef.current) {
        clearTimeout(processingTimeoutRef.current);
        processingTimeoutRef.current = null;
      }
      
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      
      // Ensure server session is terminated
      terminateServerSession().catch(e => 
        console.error("Failed to terminate server session during cleanup:", e)
      );
    };
  }, []);
  
  // Cancel ongoing processing
  const cancelTranscription = useCallback(() => {
    console.log('Cancelling ongoing transcription...');
    
    // Mark as cancelled to prevent further processing
    isCancelledRef.current = true;
    
    // Abort any ongoing fetch requests
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    
    // Clear any timeouts
    if (processingTimeoutRef.current) {
      clearTimeout(processingTimeoutRef.current);
      processingTimeoutRef.current = null;
    }
    
    // Reset state
    setIsProcessing(false);
    const newStatus: TranscriptionStatus = {
      isRecording: false,
      duration: 0,
      status: 'idle',
    };
    setStatus(newStatus);
    
    // Clear retry counter
    retryCountRef.current = 0;
    
    // Clear any error state
    setError(null);
    
    // Try to terminate the session on the server side
    terminateServerSession().catch(e => 
      console.error("Failed to terminate server session:", e)
    );
    
    // Reset cancelled flag after short delay to allow for cleanup
    setTimeout(() => {
      isCancelledRef.current = false;
    }, 500);
  }, []);
  
  // Attempt to terminate the session on the server to free resources
  const terminateServerSession = async () => {
    if (!sessionIdRef.current) return;
    
    try {
      const serverUrl = process.env.NEXT_PUBLIC_SPEECH_SERVICE_URL?.replace(/\/+$/, '') 
        || process.env.NEXT_PUBLIC_STT_SERVER_URL?.replace(/\/+$/, '') 
        || apiUrl;
      
      const transcribeUrl = `${serverUrl}/transcribe`;
      
      await fetch(transcribeUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: 'terminate',
          session_id: sessionIdRef.current
        })
      }).catch(() => {
        // Ignore errors from terminate request
      });
    } catch (error) {
      // Ignore errors, this is just a best-effort cleanup
      console.warn("Failed to terminate server session:", error);
    }
  };
  
  // Check if a transcription result is suspicious (likely an error)
  const isTranscriptionSuspicious = useCallback((text: string): boolean => {
    // Empty text is suspicious
    if (!text || text.trim().length === 0) return true;
    
    // Check against known problematic patterns
    const isSuspicious = SUSPICIOUS_PATTERNS.some(pattern => 
      text.toLowerCase().includes(pattern.toLowerCase()));
    
    // Also check if transcription is unusually short (likely an error)
    const isTooShort = text.length < 5 && text.length > 0;
    
    return isSuspicious || isTooShort;
  }, []);
  
  // Send audio to the transcription API
  const transcribeAudio = useCallback(async (audioData: string): Promise<string | null> => {
    if (!audioData) {
      throw new Error("No audio data provided");
    }
    
    // Don't proceed if cancelled
    if (isCancelledRef.current) {
      console.log("Transcription was cancelled, aborting transcribeAudio call");
      return null;
    }
    
    // Create new AbortController for this request
    if (abortControllerRef.current) {
      try {
        abortControllerRef.current.abort();
      } catch (e) {
        // Ignore errors from aborting
      }
    }
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;
    
    // Prepare URL and setup session ID
    const serverUrl = process.env.NEXT_PUBLIC_SPEECH_SERVICE_URL?.replace(/\/+$/, '') 
      || process.env.NEXT_PUBLIC_STT_SERVER_URL?.replace(/\/+$/, '') 
      || apiUrl;
    
    const transcribeUrl = `${serverUrl}/transcribe`;
    
    // Use provided session ID or the one from AudioService or generate a new one
    if (!sessionIdRef.current) {
      sessionIdRef.current = audioService.getCurrentSessionId() || crypto.randomUUID();
      console.log(`Using session ID: ${sessionIdRef.current}`);
    }
    
    try {
      console.log(`Sending audio data (${audioData.length} bytes) to ${transcribeUrl}`);
      
      // Check again if cancelled before sending
      if (isCancelledRef.current) {
        console.log("Transcription was cancelled before fetch, aborting");
        return null;
      }
      
      // Send the audio data for transcription
      const response = await fetch(transcribeUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio: audioData,
          session_id: sessionIdRef.current,
          maintain_context: false, // Disable context to avoid common errors
          high_quality: useHighQualityProcessing
        }),
        signal
      });
      
      // Check if cancelled during fetch
      if (isCancelledRef.current || signal.aborted) {
        console.log("Transcription was cancelled during fetch, aborting");
        return null;
      }
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      
      // Check if cancelled during response parsing
      if (isCancelledRef.current) {
        console.log("Transcription was cancelled during response parsing, aborting");
        return null;
      }
      
      // Check for direct transcription response
      if (result.transcription) {
        const transcriptionText = result.transcription;
        console.log(`Received transcription (${transcriptionText.length} chars)`);
        
        // Check for suspicious patterns
        if (isTranscriptionSuspicious(transcriptionText) && retryCountRef.current < maxRetries) {
          console.warn(`Detected suspicious transcription (${retryCountRef.current + 1}/${maxRetries}): "${transcriptionText}"`);
          retryCountRef.current++;
          return null; // Signal to retry
        }
        
        return transcriptionText;
      }
      // Handle processing acknowledgment - get final result
      else if (result.status === "ok" || result.status === "processing") {
        console.log("Server processing audio, requesting final transcription...");
        
        // Only proceed if not cancelled or aborted
        if (isCancelledRef.current || signal.aborted) {
          console.log("Transcription was cancelled or aborted before finish request");
          return null;
        }
        
        // Wait 1 second for processing
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Check again if cancelled during wait
        if (isCancelledRef.current) {
          console.log("Transcription was cancelled during wait, aborting");
          return null;
        }
        
        // Send 'finish' command to get final result
        const finishResponse = await fetch(transcribeUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            command: 'finish',
            session_id: sessionIdRef.current,
            maintain_context: false
          }),
          signal
        });
        
        // Check if cancelled during finish fetch
        if (isCancelledRef.current || signal.aborted) {
          console.log("Transcription was cancelled during finish fetch, aborting");
          return null;
        }
        
        if (!finishResponse.ok) {
          throw new Error(`HTTP error on finish request! status: ${finishResponse.status}`);
        }
        
        const finishResult = await finishResponse.json();
        
        // Check if cancelled during finish response parsing
        if (isCancelledRef.current) {
          console.log("Transcription was cancelled during finish response parsing, aborting");
          return null;
        }
        
        if (finishResult.transcription) {
          const transcriptionText = finishResult.transcription;
          
          // Check for suspicious patterns one more time
          if (isTranscriptionSuspicious(transcriptionText) && retryCountRef.current < maxRetries) {
            console.warn(`Detected suspicious transcription in finish response (${retryCountRef.current + 1}/${maxRetries}): "${transcriptionText}"`);
            retryCountRef.current++;
            return null; // Signal to retry
          }
          
          return transcriptionText;
        }
      }
      
      // No valid transcription found
      return null;
      
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        console.log('Transcription fetch was aborted');
        return null;
      }
      
      // Check if cancelled during error handling
      if (isCancelledRef.current) {
        console.log("Transcription was cancelled during error handling, aborting");
        return null;
      }
      
      throw err;
    }
  }, [apiUrl, isTranscriptionSuspicious, maxRetries, useHighQualityProcessing]);
  
  // Handle the transcription process with retries
  const startTranscription = useCallback(async (audioData?: string) => {
    // If audioData is not provided, try to get it from AudioService
    if (!audioData) {
      // Get processed audio from AudioService
      const buffer = audioService.getAudioBuffer();
      const processedData = audioService.processAudioBuffer(buffer);
      
      if (!processedData) {
        setError("No audio data provided or available");
        return;
      }
      
      // Convert to base64
      audioData = audioService.convertToBase64(processedData);
      
      // If still no audio data, try fallback to last recorded audio
      if (!audioData) {
        audioData = audioService.getLastAudioData();
        
        if (!audioData) {
          setError("No audio data available");
          return;
        }
      }
    }
    
    // Reset cancelled flag
    isCancelledRef.current = false;
    
    // Save audio data for potential retries
    lastAudioDataRef.current = audioData;
    
    // Reset state for new transcription
    setIsProcessing(true);
    setError(null);
    retryCountRef.current = 0;
    
    // Update status
    const newStatus: TranscriptionStatus = {
      isRecording: false,
      duration: 0,
      status: 'processing',
    };
    setStatus(newStatus);
    
    // Set timeout to handle stalled transcription
    if (processingTimeoutRef.current) {
      clearTimeout(processingTimeoutRef.current);
    }
    
    processingTimeoutRef.current = setTimeout(() => {
      // Only proceed if we're still processing and not cancelled
      if (isProcessing && !isCancelledRef.current) {
        console.warn(`Transcription timeout reached after ${timeoutMs}ms`);
        
        // Auto retry on timeout if we haven't maxed out retries
        if (retryCountRef.current < maxRetries) {
          console.log(`Auto-retrying (${retryCountRef.current + 1}/${maxRetries})...`);
          retryTranscription();
        } else {
          setError(`Transcription timed out after ${maxRetries} attempts`);
          setIsProcessing(false);
          setStatus({
            isRecording: false,
            duration: 0,
            status: 'error',
          });
        }
      }
    }, timeoutMs);
    
    try {
      // Start transcription process
      const result = await transcribeAudio(audioData);
      
      // Check if cancelled during transcription
      if (isCancelledRef.current) {
        console.log("Transcription was cancelled after transcribeAudio, aborting processing");
        return;
      }
      
      // Handle result (or retry)
      if (result) {
        // We got a valid transcription
        setTranscription(result);
        setIsProcessing(false);
        setStatus({
          isRecording: false,
          duration: 0,
          status: 'done',
        });
        
        // Clear timeout
        if (processingTimeoutRef.current) {
          clearTimeout(processingTimeoutRef.current);
          processingTimeoutRef.current = null;
        }
      } else if (retryCountRef.current < maxRetries && !isCancelledRef.current) {
        // Auto retry if we got a suspicious result
        console.log(`Automatically retrying transcription (${retryCountRef.current}/${maxRetries})...`);
        setTimeout(() => {
          if (!isCancelledRef.current) {
            retryTranscription();
          }
        }, 500);
      } else if (!isCancelledRef.current) {
        // We've exhausted our retries
        setError(`Failed to get valid transcription after ${maxRetries} attempts`);
        setIsProcessing(false);
        setStatus({
          isRecording: false,
          duration: 0,
          status: 'error',
        });
      }
    } catch (error) {
      // Don't update state if cancelled
      if (isCancelledRef.current) {
        console.log("Transcription was cancelled during error handling in startTranscription");
        return;
      }
      
      console.error("Error in transcription:", error);
      setError(error instanceof Error ? error.message : 'Transcription failed');
      setIsProcessing(false);
      setStatus({
        isRecording: false,
        duration: 0,
        status: 'error',
      });
    }
  }, [isProcessing, maxRetries, timeoutMs, transcribeAudio]);
  
  // Retry transcription with the last audio data
  const retryTranscription = useCallback(async () => {
    // Don't retry if cancelled
    if (isCancelledRef.current) {
      console.log("Transcription was cancelled, aborting retry");
      return;
    }
    
    if (!lastAudioDataRef.current) {
      setError("No audio data available for retry");
      return;
    }
    
    setIsProcessing(true);
    setError(null);
    
    try {
      const result = await transcribeAudio(lastAudioDataRef.current);
      
      // Don't update state if cancelled
      if (isCancelledRef.current) {
        console.log("Transcription was cancelled after retry attempt");
        return;
      }
      
      if (result) {
        setTranscription(result);
        setIsProcessing(false);
        setStatus({
          isRecording: false,
          duration: 0,
          status: 'done',
        });
        
        // Clear timeout
        if (processingTimeoutRef.current) {
          clearTimeout(processingTimeoutRef.current);
          processingTimeoutRef.current = null;
        }
      } else if (retryCountRef.current < maxRetries && !isCancelledRef.current) {
        // Try again with a slight delay
        setTimeout(() => {
          if (!isCancelledRef.current) {
            retryTranscription();
          }
        }, 500);
      } else if (!isCancelledRef.current) {
        setError(`Failed to get valid transcription after ${maxRetries} attempts`);
        setIsProcessing(false);
        setStatus({
          isRecording: false,
          duration: 0,
          status: 'error',
        });
      }
    } catch (error) {
      // Don't update state if cancelled
      if (isCancelledRef.current) {
        console.log("Transcription was cancelled during error handling in retryTranscription");
        return;
      }
      
      console.error("Error in transcription retry:", error);
      setError(error instanceof Error ? error.message : 'Transcription retry failed');
      setIsProcessing(false);
      setStatus({
        isRecording: false,
        duration: 0,
        status: 'error',
      });
    }
  }, [maxRetries, transcribeAudio]);
  
  return {
    startTranscription,
    cancelTranscription,
    retryTranscription,
    transcription,
    isProcessing,
    error,
    status,
    sessionId: sessionIdRef.current
  };
}; 