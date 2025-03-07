import { env } from 'process';

const API_BASE_URL = process.env.NEXT_PUBLIC_SPEECH_SERVICE_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
    speech: `${API_BASE_URL}/generate_speech`,
    transcribe: `${API_BASE_URL}/transcribe`,
    // Add other endpoints as needed
};

// API client configuration
export const API_CONFIG = {
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
};