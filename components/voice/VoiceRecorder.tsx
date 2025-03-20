import { useState, useEffect, useRef } from 'react';
import { Button } from "@heroui/button";
import { audioService } from '../../app/api/audio';
import { TranscriptionStatus } from '../../app/api/types';
import { motion } from "framer-motion";
import React, { useCallback } from 'react';

// Constants for error detection
const SUSPICIOUS_TRANSCRIPTION = "Thanks for watching.";
// Expanded list of known problematic patterns
const SUSPICIOUS_PATTERNS = [
  "thanks for watching", 
  "I'm sorry", 
  "thank you for watching",
  "processing your",
  "chunk",
  "please wait",
  "thank you", // Add specific pattern mentioned by user
  "thank you for using",
  "I am sorry",
  "I do not have",
  "Error during transcription", // Add pattern from the server error
  "SpeechRecognitionClient", // Add specific error term
  "context_prompt" // Add specific error term
];
const MAX_RETRIES = 3;

interface VoiceRecorderProps {
  onTranscriptionUpdate?: (text: string, sessionId?: string) => void;
  onRecordingStatusChange?: (status: TranscriptionStatus) => void;
  useGoogleSTT?: boolean;
  enableInterimResults?: boolean;
  showAudioVisualizer?: boolean;
  showLanguageSelector?: boolean;
  useHighQualityProcessing?: boolean;
}

export default function CombinedVoiceRecorder({ 
  onTranscriptionUpdate, 
  onRecordingStatusChange,
  useGoogleSTT = false,
  enableInterimResults = false,
  showAudioVisualizer = true,
  showLanguageSelector = false,
  useHighQualityProcessing = false
}: VoiceRecorderProps = {}) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [status, setStatus] = useState<TranscriptionStatus>({
    isRecording: false,
    duration: 0,
    status: 'idle',
  });
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioVisualizerRef = useRef<HTMLCanvasElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const processingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const retryCountRef = useRef(0);
  const lastAudioDataRef = useRef<string | null>(null);
  const audioBufferRef = useRef<Float32Array[]>([]);
  const audioProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const audioSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    audioService.connect();

    audioService.onTranscription((response) => {
      if (response.error) {
        setError(response.error);
        setIsProcessing(false);
        return;
      }
      
      const transcriptionText = response.text || response.transcription || '';
      console.log('Received transcription:', transcriptionText);
      
      // Enhanced suspicious transcription detection
      const isSuspicious = SUSPICIOUS_PATTERNS.some(pattern => 
        transcriptionText.toLowerCase().includes(pattern.toLowerCase()));
      
      // Also check if transcription is unusually short (likely an error)
      const isTooShort = transcriptionText.length < 5 && transcriptionText.length > 0;
      
      if ((isSuspicious || isTooShort) && retryCountRef.current < MAX_RETRIES) {
        console.warn(`Detected suspicious transcription (${retryCountRef.current + 1}/${MAX_RETRIES}): "${transcriptionText}"`);
        retryCountRef.current += 1;
        
        // Try fallback transcription
        if (lastAudioDataRef.current) {
          console.log("Attempting direct API fallback...");
          retryTranscriptionWithFallback(lastAudioDataRef.current);
          return;
        }
      }
      
      if (transcriptionText) {
        // Only use the response if it's not suspicious or we've exhausted retries
        if ((!isSuspicious && !isTooShort) || retryCountRef.current >= MAX_RETRIES) {
          // Track session ID if provided
          if (response.sessionId) {
            sessionIdRef.current = response.sessionId;
          }
          
          setTranscription(transcriptionText);
          
          // Call the callback if provided - pass session ID when available
          if (onTranscriptionUpdate) {
            onTranscriptionUpdate(transcriptionText, sessionIdRef.current || undefined);
          }
          
          // Consider processing complete when we have a valid transcription
          setIsProcessing(false);
          const newStatus: TranscriptionStatus = {
            isRecording: false,
            duration: status.duration,
            status: 'done' as 'done',
          };
          setStatus(newStatus);
          
          // Call the callback if provided
          if (onRecordingStatusChange) {
            onRecordingStatusChange(newStatus);
          }
          
          // Clear any pending timeout
          if (processingTimeoutRef.current) {
            clearTimeout(processingTimeoutRef.current);
            processingTimeoutRef.current = null;
          }
        }
      }
    });

    audioService.onError((errorMsg) => {
      setError(errorMsg);
      setIsProcessing(false);
      stopRecording();
    });

    return () => {
      stopVisualization();
      cleanupAudio();
      cancelProcessing();
      if (processingTimeoutRef.current) {
        clearTimeout(processingTimeoutRef.current);
      }
    };
  }, []);

  // Cancel ongoing processing
  const cancelProcessing = () => {
    if (abortControllerRef.current) {
      console.log('Cancelling ongoing processing...');
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    
    setIsProcessing(false);
    const newStatus: TranscriptionStatus = {
      isRecording: false,
      duration: 0,
      status: 'idle' as 'idle',
    };
    setStatus(newStatus);
    
    // Clear any previous transcription when cancelling
    setTranscription('');
    
    // Reset retry counter
    retryCountRef.current = 0;
    
    if (onRecordingStatusChange) {
      onRecordingStatusChange(newStatus);
    }
    
    if (processingTimeoutRef.current) {
      clearTimeout(processingTimeoutRef.current);
      processingTimeoutRef.current = null;
    }
    
    // Notify the parent component if needed
    if (onTranscriptionUpdate) {
      onTranscriptionUpdate('');
    }
    
    // Clear any error state
    setError('');
  };

  const cleanupAudio = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    
    if (audioProcessorRef.current) {
      try {
        audioProcessorRef.current.disconnect();
        audioProcessorRef.current = null;
      } catch (e) {
        console.warn('Error disconnecting audio processor:', e);
      }
    }
    
    if (audioSourceRef.current) {
      try {
        audioSourceRef.current.disconnect();
        audioSourceRef.current = null;
      } catch (e) {
        console.warn('Error disconnecting audio source:', e);
      }
    }
    
    if (audioContextRef.current) {
      try {
        audioContextRef.current.close();
        audioContextRef.current = null;
      } catch (e) {
        console.warn('Error closing audio context:', e);
      }
    }
  };

  // Function to retry transcription with direct API call
  const retryTranscriptionWithFallback = async (audioData: string) => {
    try {
      console.log("Attempting transcription with alternative API endpoint");
      const API_URL = process.env.NEXT_PUBLIC_API_URL || '/api';
      
      const response = await fetch(`${API_URL}/transcribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          audioData,
          sessionId: sessionIdRef.current || crypto.randomUUID(),
          // Use a different API flag to bypass normal processing
          alternativeMethod: true,
          // Add high quality flag if enabled
          highQuality: useHighQualityProcessing
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        const transcriptionText = result.text || result.transcription || '';
        
        // Double-check response against suspicious patterns
        const isSuspicious = SUSPICIOUS_PATTERNS.some(pattern => 
          transcriptionText.toLowerCase().includes(pattern.toLowerCase()));
        
        if (transcriptionText && !isSuspicious) {
          // Success - update the transcription
          setTranscription(transcriptionText);
          
          // Call the callback if provided
          if (onTranscriptionUpdate) {
            onTranscriptionUpdate(transcriptionText, result.sessionId || sessionIdRef.current);
          }
          
          // Update status
          setIsProcessing(false);
          const newStatus = {
            isRecording: false,
            duration: status.duration,
            status: 'done' as 'done',
          };
          setStatus(newStatus);
          
          // Call the callback if provided
          if (onRecordingStatusChange) {
            onRecordingStatusChange(newStatus);
          }
          
          // Clear retries
          retryCountRef.current = MAX_RETRIES;
        } else {
          // Try another fallback with raw audio if available
          if (audioBufferRef.current.length > 0) {
            console.log("Attempting direct processing of raw audio");
            processRawAudioBuffer();
          } else {
            console.warn("Alternative API returned suspicious transcription");
            
            // If we have exhausted all retries, use the best result we have
            if (retryCountRef.current >= MAX_RETRIES && transcriptionText) {
              setTranscription(transcriptionText);
              
              if (onTranscriptionUpdate) {
                onTranscriptionUpdate(transcriptionText, result.sessionId || sessionIdRef.current);
              }
              
              setIsProcessing(false);
              const newStatus = {
                isRecording: false,
                duration: status.duration,
                status: 'done' as 'done',
              };
              setStatus(newStatus);
              
              if (onRecordingStatusChange) {
                onRecordingStatusChange(newStatus);
              }
            }
          }
        }
      } else {
        // Try another approach with raw audio if available
        if (audioBufferRef.current.length > 0) {
          processRawAudioBuffer();
        }
      }
    } catch (err) {
      console.error("Error in transcription fallback:", err);
      
      // Process raw audio if available as last resort
      if (audioBufferRef.current.length > 0) {
        processRawAudioBuffer();
      }
    }
  };

  // Process audio directly with the server's /transcribe endpoint
  const processAudioDirectly = async (audioData?: string) => {
    try {
      // Create new AbortController for this request
      abortControllerRef.current = new AbortController();
      const signal = abortControllerRef.current.signal;
      
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      
      // Use the provided audio data or get it from the last recording
      const audio = audioData || lastAudioDataRef.current;
      
      if (!audio) {
        console.warn("No audio data available for direct processing");
        return;
      }
      
      console.log("Sending audio directly to /transcribe endpoint");
      
      // IMPORTANT: Send audio with maintain_context=false to avoid the context_prompt error
      const response = await fetch(`${API_URL}/transcribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          audio: audio,
          sessionId: sessionIdRef.current || crypto.randomUUID(),
          maintain_context: false // Set to false to avoid context_prompt error
        }),
        signal // Add the abort signal
      });
      
      if (response.ok) {
        const result = await response.json();
        
        // First check if there was an error
        if (result.error) {
          console.error("Transcription API error:", result.error);
          if (result.error.includes("context_prompt")) {
            console.warn("Detected context_prompt error, falling back to raw buffer processing");
            if (audioBufferRef.current.length > 0) {
              processRawAudioBuffer();
              return;
            }
          }
        }
        
        // Check if we have a transcription in the response
        if (result.transcription) {
          const transcriptionText = result.transcription;
          console.log("Received direct transcription:", transcriptionText);
          
          // Enhanced suspicious pattern check with detailed logging
          const suspiciousPattern = SUSPICIOUS_PATTERNS.find(pattern => 
            transcriptionText.toLowerCase().includes(pattern.toLowerCase()));
          
          if (suspiciousPattern) {
            console.warn(`Suspicious pattern detected: "${suspiciousPattern}" in transcription: "${transcriptionText}"`);
            if (audioBufferRef.current.length > 0) {
              processRawAudioBuffer();
              return;
            }
          }
          
          // If not suspicious or we've already tried multiple times, use it
          if (!suspiciousPattern || retryCountRef.current >= MAX_RETRIES) {
            setTranscription(transcriptionText);
            
            if (onTranscriptionUpdate) {
              onTranscriptionUpdate(transcriptionText, sessionIdRef.current || undefined);
            }
            
            setIsProcessing(false);
            const newStatus: TranscriptionStatus = {
              isRecording: false,
              duration: 0,
              status: 'done' as 'done',
            };
            setStatus(newStatus);
            
            if (onRecordingStatusChange) {
              onRecordingStatusChange(newStatus);
            }
          } else {
            // If suspicious, try using raw audio buffer
            console.warn("Suspicious transcription detected, trying raw audio processing");
            retryCountRef.current++;
            if (audioBufferRef.current.length > 0) {
              processRawAudioBuffer();
            }
          }
        }
        // If we received a status without transcription, it might be just an acknowledgment
        else if (result.status === "ok" || result.status === "processing") {
          console.log("Server acknowledged audio, waiting for transcription...");
          
          // Only proceed if not aborted
          if (signal.aborted) return;
          
          // Send a 'finish' command to get the final transcription
          // IMPORTANT: Set maintain_context to false here too
          const finishResponse = await fetch(`${API_URL}/transcribe`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              command: 'finish',
              sessionId: sessionIdRef.current || crypto.randomUUID(),
              maintain_context: false // Set to false to avoid context_prompt error
            }),
            signal // Add the abort signal
          });
          
          if (finishResponse.ok) {
            const finishResult = await finishResponse.json();
            
            if (finishResult.transcription) {
              const transcriptionText = finishResult.transcription;
              
              // Check for suspicious patterns in the finish result
              const suspiciousPattern = SUSPICIOUS_PATTERNS.find(pattern => 
                transcriptionText.toLowerCase().includes(pattern.toLowerCase()));
              
              if (suspiciousPattern && retryCountRef.current < MAX_RETRIES) {
                console.warn(`Suspicious pattern "${suspiciousPattern}" in finish transcription: "${transcriptionText}"`);
                retryCountRef.current++;
                if (audioBufferRef.current.length > 0) {
                  processRawAudioBuffer();
                  return;
                }
              }
              
              setTranscription(transcriptionText);
              
              if (onTranscriptionUpdate) {
                onTranscriptionUpdate(transcriptionText, sessionIdRef.current || undefined);
              }
              
              setIsProcessing(false);
              const newStatus: TranscriptionStatus = {
                isRecording: false,
                duration: 0,
                status: 'done' as 'done',
              };
              setStatus(newStatus);
              
              if (onRecordingStatusChange) {
                onRecordingStatusChange(newStatus);
              }
            }
          }
        }
      } else {
        console.error("Error from /transcribe endpoint:", response.statusText);
        if (audioBufferRef.current.length > 0) {
          processRawAudioBuffer();
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        console.log('Processing was cancelled by user');
        return;
      }
      
      console.error("Error in direct audio processing:", err);
      if (audioBufferRef.current.length > 0) {
        processRawAudioBuffer();
      }
    }
  };

  // Process raw audio buffer for improved transcription
  const processRawAudioBuffer = async () => {
    try {
      // Create new AbortController for this request if not exists
      if (!abortControllerRef.current) {
        abortControllerRef.current = new AbortController();
      }
      const signal = abortControllerRef.current.signal;
      
      console.log("Processing raw audio buffer");
      
      if (audioBufferRef.current.length === 0) {
        console.warn("No raw audio buffer available");
        return;
      }
      
      // Concatenate all audio chunks
      const totalLength = audioBufferRef.current.reduce((acc, chunk) => acc + chunk.length, 0);
      const concatenated = new Float32Array(totalLength);
      let offset = 0;
      
      for (const chunk of audioBufferRef.current) {
        concatenated.set(chunk, offset);
        offset += chunk.length;
      }
      
      // Process similar to scheduler implementation
      
      // Apply noise gate
      for (let i = 0; i < concatenated.length; i++) {
        if (Math.abs(concatenated[i]) < 0.005) {
          concatenated[i] = 0;
        }
      }
      
      // Find maximum absolute value for normalization
      let maxAbs = 0;
      for (let i = 0; i < concatenated.length; i++) {
        maxAbs = Math.max(maxAbs, Math.abs(concatenated[i]));
      }
      
      // Ensure we have a meaningful audio level (prevent near-silent audio)
      if (maxAbs < 0.01) {
        console.warn("Audio is too quiet, applying minimum threshold");
        maxAbs = 0.01;
      }
      
      // Apply normalization and compression for higher quality
      const threshold = 0.3;
      const ratio = 0.6;
      
      for (let i = 0; i < concatenated.length; i++) {
        // Normalize to [-1, 1] range
        let normalized = concatenated[i] / maxAbs;
        
        // Apply dynamic range compression
        if (Math.abs(normalized) > threshold) {
          const overThreshold = Math.abs(normalized) - threshold;
          const compressed = overThreshold * ratio;
          normalized = (normalized > 0) ? 
            threshold + compressed : 
            -threshold - compressed;
        }
        
        // Apply amplification - boosting the signal slightly
        concatenated[i] = normalized * 0.98;
      }
      
      // Calculate average level to ensure it meets server threshold
      let sumAbs = 0;
      for (let i = 0; i < concatenated.length; i++) {
        sumAbs += Math.abs(concatenated[i]);
      }
      const avgLevel = sumAbs / concatenated.length;
      console.log("Audio average level:", avgLevel);
      
      // Server requires level >= 0.001
      if (avgLevel < 0.001) {
        console.warn("Audio level too low for server processing, applying boost");
        // Boost the audio if it's too quiet
        const boostFactor = 0.002 / avgLevel;
        for (let i = 0; i < concatenated.length; i++) {
          concatenated[i] *= boostFactor;
        }
      }
      
      // Convert the Float32Array to Int16Array for better compatibility
      const int16Data = new Int16Array(concatenated.length);
      for (let i = 0; i < concatenated.length; i++) {
        int16Data[i] = Math.max(-32768, Math.min(32767, concatenated[i] * 32767));
      }
      
      // Convert to base64 using a more efficient approach
      let base64Data = '';
      const chunkSize = 24 * 1024; // Process in chunks to avoid stack overflow
      
      for (let i = 0; i < int16Data.length; i += chunkSize) {
        const chunk = new Uint8Array(int16Data.buffer.slice(
          i * 2, 
          Math.min((i + chunkSize) * 2, int16Data.length * 2)
        ));
        base64Data += String.fromCharCode.apply(null, Array.from(chunk));
      }
      
      base64Data = btoa(base64Data);
      
      // Use the direct transcribe endpoint with standard method (not context-based)
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      
      // IMPORTANT: Set maintain_context to false to avoid the context_prompt error
      const response = await fetch(`${API_URL}/transcribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio: base64Data,
          sessionId: sessionIdRef.current || crypto.randomUUID(),
          maintain_context: false // Explicitly set to false to avoid server error
        }),
        signal // Add the abort signal
      });
      
      if (response.ok) {
        const result = await response.json();
        
        // First check if there was an error
        if (result.error) {
          console.error("Raw buffer transcription API error:", result.error);
          setIsProcessing(false);
          setError(`Transcription failed: ${result.error}`);
          return;
        }
        
        // Check for transcription response
        if (result.transcription) {
          const transcriptionText = result.transcription;
          console.log("Received transcription:", transcriptionText);
          
          // Check for suspicious patterns
          const suspiciousPattern = SUSPICIOUS_PATTERNS.find(pattern => 
            transcriptionText.toLowerCase().includes(pattern.toLowerCase()));
          
          if (suspiciousPattern && retryCountRef.current < MAX_RETRIES) {
            console.warn(`Suspicious pattern "${suspiciousPattern}" detected in raw buffer transcription: "${transcriptionText}"`);
            retryCountRef.current++;
            return; // Don't use this result
          }
          
          setTranscription(transcriptionText);
          
          if (onTranscriptionUpdate) {
            onTranscriptionUpdate(transcriptionText, sessionIdRef.current || undefined);
          }
          
          setIsProcessing(false);
          const newStatus: TranscriptionStatus = {
            isRecording: false,
            duration: 0,
            status: 'done' as 'done',
          };
          setStatus(newStatus);
          
          if (onRecordingStatusChange) {
            onRecordingStatusChange(newStatus);
          }
        }
        // If it was just acknowledged, send finish command
        else if (result.status === "ok" || result.status === "processing") {
          console.log("Audio processed successfully, requesting final transcription");
          
          // Only proceed if not aborted
          if (signal.aborted) return;
          
          // Send finish command to get final transcription with maintain_context=false
          const finishResponse = await fetch(`${API_URL}/transcribe`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              command: 'finish',
              sessionId: sessionIdRef.current || crypto.randomUUID(),
              maintain_context: false // Set to false to avoid context_prompt error
            }),
            signal // Add the abort signal
          });
          
          if (finishResponse.ok) {
            const finishResult = await finishResponse.json();
            
            if (finishResult.transcription) {
              const transcriptionText = finishResult.transcription;
              
              // One last check for suspicious patterns
              const suspiciousPattern = SUSPICIOUS_PATTERNS.find(pattern => 
                transcriptionText.toLowerCase().includes(pattern.toLowerCase()));
              
              if (suspiciousPattern && retryCountRef.current < MAX_RETRIES) {
                console.warn(`Suspicious pattern "${suspiciousPattern}" in finish response: "${transcriptionText}"`);
                retryCountRef.current++;
                return; // Don't use this result
              }
              
              setTranscription(transcriptionText);
              
              if (onTranscriptionUpdate) {
                onTranscriptionUpdate(transcriptionText, sessionIdRef.current || undefined);
              }
              
              setIsProcessing(false);
              const newStatus: TranscriptionStatus = {
                isRecording: false,
                duration: 0,
                status: 'done' as 'done',
              };
              setStatus(newStatus);
              
              if (onRecordingStatusChange) {
                onRecordingStatusChange(newStatus);
              }
            }
          }
        }
      } else {
        console.error("Error from /transcribe endpoint:", response.statusText);
        setError(`Server error: ${response.statusText}`);
        setIsProcessing(false);
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        console.log('Processing was cancelled by user');
        return;
      }
      
      console.error("Error processing raw audio buffer:", err);
      setError("Error processing audio: " + (err instanceof Error ? err.message : String(err)));
      setIsProcessing(false);
    }
  };

  const startRecording = async () => {
    try {
      setError(null);
      // Reset state
      retryCountRef.current = 0;
      lastAudioDataRef.current = null;
      audioBufferRef.current = [];
      
      // Clean up any existing audio processing
      cleanupAudio();
      
      // Generate new session ID for this recording
      sessionIdRef.current = crypto.randomUUID();
      console.log(`Started new recording session: ${sessionIdRef.current}`);
      
      // Request microphone access with improved audio settings
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
          sampleRate: 16000
        } 
      });
      
      if (showAudioVisualizer) {
        startVisualization(stream);
      }
      
      // Set up advanced audio processing
      const audioContext = new AudioContext({
        sampleRate: 16000,
      });
      audioContextRef.current = audioContext;
      
      const source = audioContext.createMediaStreamSource(stream);
      audioSourceRef.current = source;
      
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      audioProcessorRef.current = processor;
      
      source.connect(processor);
      processor.connect(audioContext.destination);
      
      // Process audio data with improved quality
      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Create a copy of the data to avoid reference issues
        const dataCopy = new Float32Array(inputData.length);
        dataCopy.set(inputData);
        
        // Apply a noise gate to reduce background noise
        for (let i = 0; i < dataCopy.length; i++) {
          // Improved noise gate threshold for better signal
          if (Math.abs(dataCopy[i]) < 0.005) {
            dataCopy[i] = 0;
          }
        }
        
        audioBufferRef.current.push(dataCopy);
      };
      
      // Start transcription service
      const recorder = await audioService.startTranscription(stream);
      mediaRecorderRef.current = recorder;
      
      setIsRecording(true);
      const newStatus: TranscriptionStatus = {
        isRecording: true,
        duration: 0,
        status: 'recording' as 'recording',
      };
      setStatus(newStatus);
      
      // Call the callback if provided
      if (onRecordingStatusChange) {
        onRecordingStatusChange(newStatus);
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to start recording');
      stopVisualization();
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      
      // Store the last audio data for retry attempts
      lastAudioDataRef.current = audioService.getLastAudioData();
      
      mediaRecorderRef.current = null;
      
      setIsRecording(false);
      setIsProcessing(true);
      const newStatus: TranscriptionStatus = {
        isRecording: false,
        duration: 0,
        status: 'processing' as 'processing',
      };
      setStatus(newStatus);
      
      // Call the callback if provided
      if (onRecordingStatusChange) {
        onRecordingStatusChange(newStatus);
      }
      
      // Process the audio directly instead of relying on WebSockets
      processAudioDirectly();
      
      // Set a longer timeout to ensure we don't get stuck in processing state
      if (processingTimeoutRef.current) {
        clearTimeout(processingTimeoutRef.current);
      }
      
      processingTimeoutRef.current = setTimeout(() => {
        if (isProcessing) {
          console.log('Processing timeout reached, trying fallback...');
          
          // If we have audio data, try to process it directly
          if (lastAudioDataRef.current) {
            // Try with raw processing on timeout
            processRawAudioBuffer();
          } else if (audioBufferRef.current.length > 0) {
            processRawAudioBuffer();
          } else {
            setIsProcessing(false);
            setError('Transcription timed out. Please try again.');
          }
        }
      }, 12000); // Increase to 12 seconds timeout to allow for more processing time
    }
    
    stopVisualization();
  };

  const startVisualization = (stream: MediaStream) => {
    if (!audioVisualizerRef.current) return;
    
    try {
      // Use standard AudioContext without fallback to fix TypeScript error
      const audioCtx = new AudioContext();
      audioContextRef.current = audioCtx;
      
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.8;
      analyserRef.current = analyser;
      
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);
      
      drawVisualization();
    } catch (error) {
      console.error('Error starting visualization:', error);
    }
  };

  const drawVisualization = () => {
    if (!audioVisualizerRef.current || !analyserRef.current) return;
    
    const canvas = audioVisualizerRef.current;
    const canvasCtx = canvas.getContext('2d');
    if (!canvasCtx) return;
    
    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      if (!isRecording) return;
      animationFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      
      const barWidth = (canvas.width / bufferLength) * 2.5;
      let x = 0;
      
      const gradient = canvasCtx.createLinearGradient(0, 0, 0, canvas.height);
      gradient.addColorStop(0, '#4f46e5');
      gradient.addColorStop(1, '#818cf8');
      
      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height;
        canvasCtx.fillStyle = gradient;
        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        x += barWidth + 1;
      }
    };
    
    draw();
  };

  const stopVisualization = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    if (audioVisualizerRef.current) {
      const canvasCtx = audioVisualizerRef.current.getContext('2d');
      if (canvasCtx) {
        canvasCtx.clearRect(0, 0, audioVisualizerRef.current.width, audioVisualizerRef.current.height);
      }
    }
  };

  const resetRecorder = () => {
    setTranscription('');
    sessionIdRef.current = null;
    retryCountRef.current = 0;
    lastAudioDataRef.current = null;
    audioBufferRef.current = [];
    
    const newStatus: TranscriptionStatus = {
      isRecording: false,
      duration: 0,
      status: 'idle' as 'idle',
    };
    setStatus(newStatus);
    
    // Call the callback if provided
    if (onRecordingStatusChange) {
      onRecordingStatusChange(newStatus);
    }
    
    // Clear transcription in parent component if callback provided
    if (onTranscriptionUpdate) {
      onTranscriptionUpdate('');
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '8px',
        width: '100%',
      }}
    >
      {error && (
        <p style={{ color: '#d32f2f', margin: 0, fontSize: '0.875rem' }}>
          {error}
        </p>
      )}

      <div style={{ position: 'relative', display: 'inline-flex' }}>
        <button
          style={{
            width: '80px',
            height: '80px',
            border: '2px solid',
            borderColor: status.isRecording ? '#9c27b0' : '#1976d2',
            borderRadius: '50%',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            backgroundColor: 'transparent',
            cursor: isProcessing ? 'not-allowed' : 'pointer',
            opacity: isProcessing ? 0.7 : 1,
          }}
          onClick={status.isRecording ? stopRecording : startRecording}
          disabled={isProcessing}
        >
          {status.isRecording ? (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="6" y="6" width="12" height="12" fill="currentColor" />
            </svg>
          ) : (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" fill="currentColor" />
              <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" fill="currentColor" />
            </svg>
          )}
        </button>
        {isProcessing && (
          <div
            style={{
              position: 'absolute',
              top: '-10px',
              left: '-10px',
              width: '100px',
              height: '100px',
              border: '2px solid #1976d2',
              borderRadius: '50%',
              borderTopColor: 'transparent',
              animation: 'spin 1s linear infinite',
              zIndex: -1,
            }}
          />
        )}
      </div>
      
      {isProcessing && (
        <Button 
          color="danger" 
          onClick={cancelProcessing}
          className="mt-2 border border-danger-200"
          size="sm"
          startContent={
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          }
        >
          Cancel
        </Button>
      )}

      <p style={{ margin: 0, color: '#666', fontSize: '0.875rem' }}>
        {status.isRecording
          ? `Recording... ${Math.floor(status.duration)} seconds`
          : isProcessing
          ? 'Processing audio...'
          : status.status === 'done'
          ? 'Transcription completed'
          : 'Click to record'}
      </p>
      
      {transcription && (
        <div
          style={{
            marginTop: '16px',
            padding: '16px',
            width: '100%',
            backgroundColor: 'rgba(0, 0, 0, 0.05)',
            borderRadius: '4px',
            maxHeight: '200px',
            overflowY: 'auto',
          }}
        >
          <p style={{ margin: 0 }}>{transcription}</p>
        </div>
      )}
      
      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}