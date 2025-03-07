# MinoHealth CRM: Technical Documentation

## System Architecture

The MinoHealth CRM system is built with a modern microservices architecture, separating concerns between frontend user interfaces, backend business logic, and specialized AI services for speech processing.

### Architecture Diagram

```
┌─────────────────┐     ┌────────────────────┐     ┌─────────────────────┐
│                 │     │                    │     │                     │
│  Next.js        │◄───►│  FastAPI           │◄───►│  Speech-to-Text     │
│  Frontend       │     │  Backend Service   │     │  Service (Whisper)  │
│                 │     │                    │     │                     │
└─────────────────┘     └────────────────────┘     └─────────────────────┘
                                 ▲                           ▲
                                 │                           │
                                 ▼                           ▼
                        ┌────────────────────┐     ┌─────────────────────┐
                        │                    │     │                     │
                        │  Database          │     │  Text-to-Speech     │
                        │  (SQLite)          │     │  Service (YarnGPT)  │
                        │                    │     │                     │
                        └────────────────────┘     └─────────────────────┘
```

### Components

#### 1. Frontend (Next.js)

The frontend is built using Next.js, a React framework that provides server-side rendering, static site generation, and an optimized developer experience. Key frontend components include:

- **Page Components**: Application pages like the appointment scheduler
- **Voice Recorder**: Component for capturing audio and displaying visualizations
- **WebSocket Client**: Real-time communication with the backend
- **API Services**: TypeScript clients that interface with the backend APIs

#### 2. Backend Service (FastAPI)

The backend is built using FastAPI, a modern, high-performance Python web framework. It handles:

- **API Endpoints**: RESTful HTTP endpoints
- **WebSocket Handlers**: Real-time communication channels
- **Business Logic**: Appointment scheduling and management
- **Data Processing**: Managing and transforming data between services

#### 3. Speech-to-Text Service

A specialized service built on the Whisper model that provides accurate speech recognition:

- **Audio Processing**: Processes audio chunks from the WebSocket stream
- **Transcription Engine**: Uses Whisper large model for accurate transcription
- **Noise Filtering**: Filters out background noise for clearer recognition
- **Context Awareness**: Uses previous transcriptions to improve accuracy

#### 4. Text-to-Speech Service

A service based on YarnGPT2b that generates natural-sounding speech:

- **Text Processing**: Prepares text for speech synthesis
- **Voice Selection**: Multiple voice options with different characteristics
- **Audio Generation**: Creates high-quality audio from text input
- **Audio Encoding**: Encodes audio for efficient transfer over HTTP

#### 5. Database

SQLite database for storing:

- **Patient Records**: Information about patients
- **Appointments**: Scheduled appointments and their details
- **Doctor Information**: Available doctors and their specialties
- **Available Slots**: Time slots for scheduling appointments

## Key Technologies

### Backend Technologies

- **Python 3.9+**: Core programming language
- **FastAPI**: Web framework for building APIs
- **Uvicorn**: ASGI server for running the FastAPI application
- **WebSockets**: For real-time communication
- **PyAudio**: For audio capture and processing
- **NumPy**: For efficient numerical operations on audio data
- **Whisper**: AI model for speech recognition
- **YarnGPT2b**: AI model for speech synthesis
- **SQLite**: Lightweight database for data storage

### Frontend Technologies

- **TypeScript**: Typed JavaScript for better development experience
- **Next.js**: React framework with optimized rendering
- **React**: UI component library
- **WebSockets**: For real-time communication with the backend
- **Web Audio API**: For audio visualization and processing
- **MediaRecorder API**: For capturing audio from the browser
- **Tailwind CSS**: For styling components

## Data Flow

1. **Audio Capture**: 
   - User speaks into the microphone
   - Audio is captured using the MediaRecorder API
   - Audio visualization is shown using Web Audio API

2. **Speech-to-Text Processing**:
   - Audio is sent to the backend via WebSocket
   - Backend forwards audio to the STT service
   - STT service transcribes audio to text
   - Transcription is returned to frontend

3. **Appointment Scheduling**:
   - Frontend sends transcription to backend for processing
   - Backend extracts appointment details from transcription
   - Backend checks doctor availability
   - Backend creates appointment in database
   - Confirmation is sent back to frontend

4. **Text-to-Speech Response**:
   - Backend generates confirmation message
   - Message is sent to TTS service
   - TTS service generates audio response
   - Audio is streamed back to frontend
   - Frontend plays the audio response

## Code Structure

### Backend

```
backend/
├── __init__.py               # Package initialization
├── api.py                    # Main API endpoints and WebSocket handlers
├── appointment_scheduler.py  # Appointment scheduling logic
├── config.py                 # Configuration settings
├── database.py               # Database connection and models
├── scheduler.py              # Scheduling assistant logic
├── STT_client.py             # Speech-to-Text client
├── STT_server.py             # Speech-to-Text server implementation
├── TTS_client.py             # Text-to-Speech client
├── TTS_server.py             # Text-to-Speech server implementation
├── requirements.txt          # Python dependencies
└── prompt.json               # AI prompt templates
```

### Frontend

```
app/
├── api/                    # API clients
│   ├── audio.ts            # Audio service for STT/TTS
│   ├── types.ts            # TypeScript type definitions
│   └── ws/                 # WebSocket handlers
├── appointment-manager/    # Appointment management UI
├── appointment-scheduler/  # Scheduling UI
└── page.tsx                # Main application page

components/
├── voice/                  # Voice-related components
│   └── VoiceRecorder.tsx   # Audio recording component
└── ...                     # Other UI components
```

## Security Considerations

- **Authentication**: Simple API key authentication for WebSocket connections
- **Input Validation**: All inputs are validated on both client and server
- **Error Handling**: Comprehensive error handling to prevent information leakage
- **Data Protection**: All data is stored securely in the database
- **Secure Communication**: WebSocket connections can be secured with TLS

## Performance Optimization

- **Audio Chunking**: Audio is processed in manageable chunks
- **Streaming Responses**: Text and audio responses are streamed for faster feedback
- **Caching**: Frequently accessed data is cached for improved response times
- **Connection Management**: WebSocket connections are managed efficiently

## Deployment Considerations

- **Environment Configuration**: Using .env files for configuration
- **Containerization**: Can be containerized with Docker
- **Scaling**: Each service can be scaled independently
- **Monitoring**: Logging is implemented for monitoring and debugging

---

This document provides an overview of the technical architecture and implementation of the MinoHealth CRM system. For API-specific documentation, please see the API Documentation.