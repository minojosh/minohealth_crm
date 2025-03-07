# MinoHealth CRM

A modern healthcare management system with advanced speech recognition capabilities for appointment scheduling and patient management.

## Overview

MinoHealth CRM is an integrated healthcare management platform that uses AI-powered speech recognition to streamline the appointment scheduling process. The system combines a modern web frontend with a powerful backend that processes speech, schedules appointments, and manages patient data.

## Features

- **Speech-to-Text Transcription**: Record voice input and convert it to text using a custom speech recognition service
- **Voice-Driven Appointment Scheduling**: Schedule appointments using natural voice commands
- **Patient Information Management**: Store and retrieve patient data
- **Doctor Availability**: View available time slots for doctors
- **Real-time Audio Visualization**: Visual feedback during voice recording
- **Multi-Platform Support**: Works on desktop and mobile browsers

## Technology Stack

- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **Backend**: FastAPI (Python), SQLite
- **Speech Processing**: Custom STT (Speech-to-Text) server using Whisper model
- **Text-to-Speech**: YarnGPT2b model for natural speech synthesis
- **WebSockets**: Real-time communication between frontend and backend
- **Environment**: Configurable via .env files

## Getting Started

### Prerequisites

- Node.js (v16+)
- Python (v3.9+)
- PyAudio (for audio processing)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/minohealth-crm.git
   cd minohealth-crm
   ```

2. Install frontend dependencies:
   ```bash
   npm install
   ```

3. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your configuration:
   ```
   # STT (Speech-to-Text) Server Configuration
   STT_SERVER_URL=https://your-stt-server-url
   ```

### Running the Application

1. Start the backend server:
   ```bash
   cd backend
   python api.py
   ```

2. In a separate terminal, start the frontend development server:
   ```bash
   npm run dev
   ```

3. Open your browser and navigate to `http://localhost:3000`

## Architecture

The system is built with a microservices architecture:

1. **Frontend Service**: Next.js application that provides the user interface
2. **Backend API**: FastAPI application that handles business logic and data processing
3. **Speech-to-Text Service**: Specialized service for audio transcription
4. **Text-to-Speech Service**: Service that converts text to natural-sounding speech

## Project Structure

```
minohealth_crm/
├── app/                     # Next.js app directory
│   ├── api/                 # Frontend API clients
│   ├── appointment-manager/ # Appointment management interface
│   ├── appointment-scheduler/ # Scheduling interface
│   ├── data-extractor/      # Data extraction tools
│   └── reports/             # Reporting dashboard
├── backend/                 # Python backend
│   ├── STT_client.py        # Speech-to-text client
│   ├── STT_server.py        # Speech-to-text server
│   ├── TTS_client.py        # Text-to-speech client
│   ├── TTS_server.py        # Text-to-speech server
│   ├── api.py               # Main API endpoints
│   ├── scheduler.py         # Appointment scheduling logic
│   └── appointment_scheduler.py # Appointment management
├── components/              # React components
│   └── voice/               # Voice-related components
└── public/                  # Static assets
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or support, please contact [joshua@minohealth.org](mailto:[joshua@minohealth.org).

---

© 2023 MinoHealth AI Labs. All rights reserved.
