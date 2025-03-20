import { env } from 'process';

const API_BASE_URL = process.env.NEXT_PUBLIC_SPEECH_SERVER_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
    // Speech synthesis endpoints
    tts: `${API_BASE_URL}/tts`,
    ttsStream: `${API_BASE_URL}/tts-stream`,
    generateSpeech: `${API_BASE_URL}/generate_speech`,
    // Speech-to-text endpoints
    transcribe: `${API_BASE_URL}/transcribe`,
    // WebSocket endpoints
    wsAudio: `${API_BASE_URL}/ws/audio`,
    wsSchedule: `${API_BASE_URL}/ws/schedule`,
    wsConversation: `${API_BASE_URL}/ws/conversation`,
};

// API client configuration
export const API_CONFIG = {
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
    websocketAuth: {
        token: 'audio_access_token_2023'
    }
};