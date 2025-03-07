# MinoHealth CRM: API Documentation

This document outlines the API endpoints available in the MinoHealth CRM system, including both HTTP endpoints and WebSocket connections.

## Base URLs

- **HTTP API Base URL**: `http://localhost:8000`
- **WebSocket Base URL**: `ws://localhost:8000`

In production, replace `localhost:8000` with your actual domain name.

## Authentication

Most endpoints require an API key for authentication, which should be included as a query parameter:

```
?token=audio_access_token_2023
```

## HTTP Endpoints

### GET `/`

Root endpoint to verify API is running.

#### Response

```json
{
  "message": "Welcome to Moremi AI Scheduler API"
}
```

### POST `/transcribe`

Transcribes audio data to text.

#### Request Body

```json
{
  "audio": "base64_encoded_audio_data"
}
```

OR (to signal completion):

```json
{
  "command": "finish"
}
```

#### Response

```json
{
  "transcription": "Transcribed text here"
}
```

OR (if error):

```json
{
  "error": "Error message"
}
```

## WebSocket Endpoints

### WebSocket `/ws/audio`

Real-time audio transcription endpoint.

#### Connection

Connect to this endpoint to establish a WebSocket connection for audio processing:

```
ws://localhost:8000/ws/audio?token=audio_access_token_2023
```

#### Client-to-Server Messages

1. **Audio Transcription Request**:

```json
{
  "type": "transcription",
  "payload": {
    "audioData": "base64_encoded_audio_data",
    "sampleRate": 16000,
    "processComplete": false
  },
  "timestamp": 1685123456789
}
```

#### Server-to-Client Messages

1. **Transcription Response**:

```json
{
  "type": "transcription",
  "payload": {
    "text": "Transcribed text here",
    "transcription": "Transcribed text here", 
    "confidence": 0.9,
    "isComplete": false
  },
  "timestamp": 1685123456789
}
```

2. **Error Response**:

```json
{
  "type": "error",
  "payload": {
    "message": "Error message"
  },
  "timestamp": 1685123456789
}
```

3. **Status Response**:

```json
{
  "type": "status",
  "payload": {
    "message": "Status message"
  },
  "timestamp": 1685123456789
}
```

### WebSocket `/ws/schedule/{patient_id}`

Endpoint for handling the appointment scheduling flow for a specific patient.

#### Connection

Connect to this endpoint to establish a WebSocket connection for scheduling:

```
ws://localhost:8000/ws/schedule/123
```

Where `123` is the patient_id.

#### Client-to-Server Messages

1. **Context Initialization**:

```json
{
  "context": "Patient context information here"
}
```

2. **Start Listening Command**:

```json
{
  "command": "start_listening"
}
```

3. **Finish Command**:

```json
{
  "command": "finish"
}
```

#### Server-to-Client Messages

1. **Initial Greeting**:

```json
{
  "type": "message",
  "text": "Welcome to Moremi AI Scheduler! Please send the conversation context to begin."
}
```

2. **Context Acknowledgment**:

```json
{
  "type": "message",
  "text": "Context received. I am listening..."
}
```

3. **Audio Response**:

```json
{
  "type": "audio",
  "audio": "base64_encoded_audio_data",
  "sample_rate": 24000
}
```

4. **Transcription Response**:

```json
{
  "type": "transcription",
  "text": "Transcribed text here"
}
```

5. **Assistant Response**:

```json
{
  "type": "message",
  "text": "Assistant response text here"
}
```

6. **Doctor List Response**:

```json
{
  "type": "doctors",
  "data": [
    {
      "doctor_id": 1,
      "doctor_name": "Dr. John Smith",
      "specialty": "Cardiology",
      "available_slots": [
        "2023-07-10 10:00:00",
        "2023-07-10 11:00:00",
        "2023-07-11 09:00:00"
      ]
    }
  ]
}
```

7. **Conversation Summary**:

```json
{
  "type": "summary",
  "text": "Summary of the conversation and scheduled appointment"
}
```

8. **Appointment Confirmation**:

```json
{
  "type": "appointment",
  "status": "success",
  "data": {
    "doctor_id": 1,
    "patient_id": 123,
    "appointment_date": "2023-07-10",
    "appointment_time": "10:00:00",
    "reason": "Checkup"
  }
}
```

9. **Error Response**:

```json
{
  "type": "error",
  "message": "Error message"
}
```

## Frontend API Service (TypeScript)

The frontend includes a TypeScript service to interact with these endpoints:

### Audio Service

```typescript
// Import the service
import { audioService } from '../../app/api/audio';

// Connect to WebSocket
audioService.connect();

// Start speech recognition with a media stream
const mediaRecorder = await audioService.startTranscription(stream);

// Handle transcription results
audioService.onTranscription((response) => {
  console.log('Transcription:', response.text);
});

// Handle errors
audioService.onError((errorMsg) => {
  console.error('Error:', errorMsg);
});

// Generate speech from text
const ttsResponse = await audioService.synthesizeSpeech("Hello, how can I help you?");
```

## Data Types

### AudioTranscriptionRequest

```typescript
interface AudioTranscriptionRequest {
  audioData: string; // Base64 encoded audio data
  language?: string; // Optional language code
  processComplete?: boolean; // Flag to indicate this is a complete recording (not a streaming chunk)
}
```

### AudioTranscriptionResponse

```typescript
interface AudioTranscriptionResponse {
  text: string;
  confidence?: number;
  segments?: TranscriptionSegment[];
  error?: string;
  isComplete?: boolean; // Flag to indicate if this is the final transcription
}
```

### TextToSpeechRequest

```typescript
interface TextToSpeechRequest {
  text: string;
  speaker?: string; // Voice/speaker to use
  speed?: number; // Speech rate
}
```

### TextToSpeechResponse

```typescript
interface TextToSpeechResponse {
  audioData: string; // Base64 encoded audio data
  duration?: number; // Duration in seconds
  error?: string;
}
```

## Error Codes

| Code | Description                   |
|------|-------------------------------|
| 400  | Bad Request                   |
| 401  | Unauthorized                  |
| 403  | Forbidden                     |
| 404  | Not Found                     |
| 500  | Internal Server Error         |
| 1008 | Policy Violation (WebSocket)  |

## Rate Limiting

API endpoints are subject to rate limiting:
- 100 requests per minute for HTTP endpoints
- 10 WebSocket connections per IP address

## Examples

### Example 1: Transcribing Audio

```javascript
// Establish WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws/audio?token=audio_access_token_2023');

// Send audio data
ws.send(JSON.stringify({
  type: 'transcription',
  payload: {
    audioData: 'base64_encoded_audio_data',
    sampleRate: 16000,
    processComplete: false
  },
  timestamp: Date.now()
}));

// Listen for transcription results
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.type === 'transcription') {
    console.log('Transcription:', message.payload.text);
  }
};
```

### Example 2: Scheduling an Appointment

```javascript
// Establish WebSocket connection for patient ID 123
const ws = new WebSocket('ws://localhost:8000/ws/schedule/123');

// Send context information
ws.send(JSON.stringify({
  context: "Patient John Doe needs to schedule a cardiology appointment next week."
}));

// Start listening for speech
ws.send(JSON.stringify({
  command: "start_listening"
}));

// Handle responses
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  switch (message.type) {
    case 'doctors':
      console.log('Available doctors:', message.data);
      break;
    case 'appointment':
      console.log('Appointment confirmed:', message.data);
      break;
    case 'message':
      console.log('Assistant says:', message.text);
      break;
  }
};
```

## Additional Notes

- All timestamps are in Unix milliseconds
- Audio data should be in PCM format, 16-bit, 16kHz sample rate
- WebSocket connections will time out after 5 minutes of inactivity
- Base64 encoded audio should be chunked to avoid memory issues