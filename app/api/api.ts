import { env } from 'process';
import dotenv from 'dotenv';

dotenv.config();


// Base URLs from environment variables
const SPEECH_SERVICE_BASE_URL = (process.env.NEXT_PUBLIC_SPEECH_SERVICE_URL || 'http://localhost:8000').replace(/\/+$/, ''); // Reverted to use NEXT_PUBLIC_ for frontend access
const BACKEND_API_BASE_URL = (process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000').replace(/\/+$/, ''); // Default to Next.js default port if not set

export const API_ENDPOINTS = {
    // Speech Service Endpoints
    tts: `${SPEECH_SERVICE_BASE_URL}/tts`,
    ttsStream: `${SPEECH_SERVICE_BASE_URL}/tts-stream`,
    transcribe: `${SPEECH_SERVICE_BASE_URL}/transcribe`,
    uploadTranscribe: `${SPEECH_SERVICE_BASE_URL}/upload-transcribe/`,
    wsAudio: `${SPEECH_SERVICE_BASE_URL}/ws/audio`,
    wsSchedule: `${SPEECH_SERVICE_BASE_URL}/ws/schedule`,
    wsConversation: `${SPEECH_SERVICE_BASE_URL}/ws/conversation`,

    // Backend API Endpoints (using relative paths for Next.js API routes, assuming they are served from the same origin)
    // Alternatively, use BACKEND_API_BASE_URL if the backend is separate
    medicalExtract: `${BACKEND_API_BASE_URL}/api/medical/extract`, // Example using relative path
    medicalSoap: `${BACKEND_API_BASE_URL}/api/medical/soap`,       // Example using relative path
    patients: `${BACKEND_API_BASE_URL}/api/patients`,              // Example using relative path
    patientDetails: (id: number) => `/api/patients/${id}`, // Example using relative path
    
    // Deprecated Backend Endpoints (if needed)
    deprecatedExtract: `/api/extract/data`, // Example using relative path
};

// API client configuration
export const API_CONFIG = {
    speechServiceBaseURL: SPEECH_SERVICE_BASE_URL,
    backendApiBaseURL: BACKEND_API_BASE_URL, // Can be used if calling a separate backend
    headers: {
        'Content-Type': 'application/json',
    },
    websocketAuth: {
        token: 'audio_access_token_2023'
    }
};

// Helper to construct WebSocket URLs
export const getWebSocketURL = (endpointPath: string, params: Record<string, string | number> = {}) => {
    // Use SPEECH_SERVICE_BASE_URL for WebSocket URLs
    const baseUrl = SPEECH_SERVICE_BASE_URL.replace(/^http/, 'ws');
    const url = new URL(endpointPath, baseUrl);
    
    // Add query parameters
    Object.entries(params).forEach(([key, value]) => {
        url.searchParams.append(key, String(value));
    });
    
    // Add auth token
    if (API_CONFIG.websocketAuth.token) {
        url.searchParams.append('token', API_CONFIG.websocketAuth.token);
    }
    
    return url.toString();
};