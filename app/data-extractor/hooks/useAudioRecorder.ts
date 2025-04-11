import React, { useState, useRef,useEffect, useCallback } from 'react';
import { ExtractedDataResponse, extractData, extractFromAudio } from '../api'; // Assuming api is in the same dir or adjust path
import { API_ENDPOINTS } from '../../api/api';

const AUDIO_SAMPLE_RATE = 16000;
const AUDIO_CHUNK_SIZE = 4096;

// Helper to convert Float32Array audio buffer to Base64
const audioBufferToBase64 = (buffer: Float32Array): string => {
    let binary = '';
    const bytes = new Uint8Array(buffer.buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
};

// Helper to normalize audio buffer
const normalizeAudio = (buffer: Float32Array): Float32Array | null => {
    let maxAbs = 0;
    for (let i = 0; i < buffer.length; i++) {
        maxAbs = Math.max(maxAbs, Math.abs(buffer[i]));
    }

    if (maxAbs === 0) return null; // Avoid division by zero for silent audio

    const normalized = new Float32Array(buffer.length);
    for (let i = 0; i < buffer.length; i++) {
        normalized[i] = buffer[i] / maxAbs;
    }

    // Check level after normalization (basic check)
    let sumAbs = 0;
    for (let i = 0; i < normalized.length; i++) {
        sumAbs += Math.abs(normalized[i]);
    }
    const level = sumAbs / normalized.length;
    if (level < 0.001) {
        console.warn(`Audio level too low (${level}) after normalization.`);
        return null; // Consider audio too quiet
    }

    return normalized;
};

export const useAudioRecorder = (options?: { onTranscriptionUpdate?: (text: string) => void }) => {
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [transcription, setTranscription] = useState('');
    const [extractedData, setExtractedData] = useState<ExtractedDataResponse | null>(null);

    const streamRef = useRef<MediaStream | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const processorRef = useRef<ScriptProcessorNode | null>(null);
    const audioBufferRef = useRef<Float32Array[]>([]);

    const cleanupAudioResources = useCallback(() => {
        processorRef.current?.disconnect();
        processorRef.current = null;
        audioContextRef.current?.close().catch(console.warn);
        audioContextRef.current = null;
        streamRef.current?.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        audioBufferRef.current = [];
    }, []);

    const processRecording = useCallback(async () => {
        if (audioBufferRef.current.length === 0) {
            setError('No audio data captured.');
            setIsProcessing(false);
            return;
        }

        setIsProcessing(true);
        setError(null);
        setTranscription(''); // Clear previous transcription
        setExtractedData(null);

        try {
            // Concatenate all audio chunks
            const totalLength = audioBufferRef.current.reduce((acc, chunk) => acc + chunk.length, 0);
            const concatenated = new Float32Array(totalLength);
            let offset = 0;
            audioBufferRef.current.forEach(chunk => {
                concatenated.set(chunk, offset);
                offset += chunk.length;
            });

            // Normalize audio
            const normalizedAudio = normalizeAudio(concatenated);
            if (!normalizedAudio) {
                setError('Audio signal too weak or silent.');
                setIsProcessing(false);
                cleanupAudioResources();
                return;
            }

            // Convert to Base64
            const base64Data = audioBufferToBase64(normalizedAudio);
            console.log("Base64 audio data length:", base64Data.length);

            // Send to transcription API
            const transcribeUrl = API_ENDPOINTS.transcribe; // Assuming API_ENDPOINTS is available
            const response = await fetch(transcribeUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ audio: base64Data, maintain_context: false }) // Use non-streaming
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Transcription API error: ${response.status} - ${errorText}`);
            }

            const result = await response.json();
            const receivedTranscription = result.transcription?.trim();

            if (!receivedTranscription || receivedTranscription.toLowerCase() === 'thank you.' || receivedTranscription.toLowerCase() === 'thank you') {
                 console.error("Received invalid transcription:", receivedTranscription);
                 setError('Speech recognition failed or returned an empty result.');
                 setTranscription('');
            } else {
                setTranscription(receivedTranscription);
                options?.onTranscriptionUpdate?.(receivedTranscription);

                // --- Call extractData --- 
                try {
                    console.log("Calling extractData with transcription:", receivedTranscription);
                    const extracted = await extractData(receivedTranscription);
                    setExtractedData(extracted);
                } catch (extractError) {
                    console.error('Error during data extraction:', extractError);
                    setError(`Data extraction failed: ${extractError instanceof Error ? extractError.message : String(extractError)}`);
                    setExtractedData(null); // Clear previous results on error
                }
                // ----------------------- 
            }

        } catch (err) {
            console.error('Error processing recording:', err);
            setError(`Processing failed: ${err instanceof Error ? err.message : String(err)}`);
        } finally {
            setIsProcessing(false);
            cleanupAudioResources(); // Clean up after processing
        }
    }, [cleanupAudioResources, options]);

    const startRecording = useCallback(async () => {
        if (isRecording) return;

        setError(null);
        setTranscription('');
        setExtractedData(null);
        cleanupAudioResources(); // Ensure clean state

        try {
            streamRef.current = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: AUDIO_SAMPLE_RATE,
                    channelCount: 1
                }
            });

            audioContextRef.current = new AudioContext({ sampleRate: AUDIO_SAMPLE_RATE });
            const source = audioContextRef.current.createMediaStreamSource(streamRef.current);
            processorRef.current = audioContextRef.current.createScriptProcessor(AUDIO_CHUNK_SIZE, 1, 1);

            processorRef.current.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                audioBufferRef.current.push(new Float32Array(inputData)); // Store a copy
            };

            source.connect(processorRef.current);
            processorRef.current.connect(audioContextRef.current.destination); // Connect to destination to start processing

            setIsRecording(true);

        } catch (err) {
            console.error('Error starting recording:', err);
            let message = 'Could not access microphone.';
            if (err instanceof Error) {
                if (err.name === 'NotAllowedError') message = 'Microphone access denied.';
                else if (err.name === 'NotFoundError') message = 'No microphone found.';
            }
            setError(message);
            cleanupAudioResources();
        }
    }, [isRecording, cleanupAudioResources]);

    const stopRecording = useCallback(() => {
        if (!isRecording) return;

        setIsRecording(false);
        // Stop the stream tracks *before* processing
        streamRef.current?.getTracks().forEach(track => track.stop());
        // Process the collected audio
        processRecording();
        // Cleanup happens inside processRecording or if start fails

    }, [isRecording, processRecording]);

    // Ensure cleanup on unmount
    useEffect(() => {
        return () => {
            cleanupAudioResources();
        };
    }, [cleanupAudioResources]);

    return {
        isRecording,
        isProcessing,
        error,
        transcription,
        extractedData,
        startRecording,
        stopRecording,
        setTranscription, // Expose setter for manual text input
        setError // Expose setter to clear errors
    };
}; 