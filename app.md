# MinoHealth CRM - Next.js Frontend Integration Plan

## Updated Architecture Overview

### System Architecture
```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Next.js        │      │  API Layer      │      │  Database       │
│  Frontend       │◄────►│  (Backend)      │◄────►│  (SQLite)  │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

## 1. Project Structure

```
minohealth-crm/
├── frontend/                 # Next.js application
│   ├── app/                  # App Router structure
│   │   ├── appointment manager/# Appointment management (reminders, etc)
│   │   ├── data-extractor/   # Data extraction tools
│   │   ├── reports(differential diagnosis) # Analytics and reports
│   │   └── appointment scheduler/         # Scheduling of Appointments
│   ├── components/           # Reusable UI components
│   │   ├── ui/               # General UI components
│   │   ├── voice/            # Voice recording components
│   │   ├── data-input/       # Data input components (days ahead, hours ahead, type of medication/appointment)
│   │   └── result-display/   # Show results from processes on frontend
│   ├── lib/                  # Utility functions & API clients
│   │   ├── api/              # API client functions
│   │   ├── audio/            # Audio processing utilities
│   │   └── transcription/    # Transcription processing
│   └── public/               # Static assets
└── backend/                  # Existing MinoHealth CRM backend
```

## 2. Technology Stack

### Frontend
- **Framework**: Next.js 14+ with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS with HeroUI
- **State Management**: React Context API + SWR/React Query
- **Authentication**: NextAuth.js (Auth.js)
- **Charts/Visualization**: Recharts or Chart.js
- **Form Handling**: React Hook Form + Zod
- **Voice Processing**: Web Audio API
- **WebSockets**: Socket.IO for real-time transcription

### Backend Integration
- FAST API endpoints
- WebSocket connections for real-time features
- Authentication via JWT tokens
- Colab enabled endpoints for STT and TTS

## 3. Development Process & Timeline

1. **Project Setup & Core Components** (Week 1)
   - Initialize Next.js project with TypeScript
   - Setup authentication infrastructure
   - Create layout components (Header, Sidebar, Footer)

3. **Appointment Scheduling** (Week 3)
   - Build audio transcription service
   - parse transcript to moremi

4. **Voice-to-Text Data Extractor** (Week 4)
   - Create voice recording components
   - Implement real-time transcription UI
   - Build data visualization and extraction interfaces

5. **Reports & Analytics** (Week 5)
   - Develop reporting dashboard
   - Create data visualization components

6. **Settings & System Configuration** (Week 6)
   - User profile management
   - System preferences

7. **Testing, Optimization & Deployment** (Weeks 7-8)
   - Comprehensive testing
   - Performance optimization
   - Deployment configuration



## 5. Voice Input Processing Pipeline

### Voice Recording & Data Extraction Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Voice          │     │  Azure          │     │  Transcript     │     │  Moremi         │     │  Structured     │
│  Recording      │────►│  Speech SDK     │────►│  Processing     │────►│  API            │────►│  YAML Data      │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Components of the Voice Processing Pipeline

1. **Voice Recording Component**
   - Microphone input handling
   - Audio visualization (waveform/volume levels)
   - Recording controls (start, stop, pause)
   - Audio storage in browser

2. **Real-time Transcription System**
   - WebSocket connection to transcription service
   - Live display of transcription
   - Speaker diarization (identifying different speakers)
   - Transcription history management

3. **Data Extraction Processing**
   - Submission to Moremi API
   - Parsing and validation of extracted data
   - Error handling and retry logic

4. **Result Display and Management**
   - YAML data visualization
   - Data editing and correction interface
   - Patient record linkage
   - Storage and archiving


## 7. Deployment Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Code       │     │  Build &    │     │  Testing    │     │  Production │
│  Repository │────►│  CI/CD      │────►│  Environment│────►│  Deployment │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

1. **Development**:
   - Local development using Next.js dev server
   - Backend runs locally or connects to dev API

2. **CI/CD Pipeline**:
   - GitHub Actions or similar CI/CD platform
   - Automated testing on pull requests
   - Build and deployment to staging on merge to main

3. **Production Deployment**:
   - Static export or server-side rendering options
   - Deploy to Vercel, Netlify, or custom server
   - Environment configuration for production API endpoints
