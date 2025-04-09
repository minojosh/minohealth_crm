# Project Documentation: MinoHealth CRM

This project appears to be a web application focused on healthcare, likely a CRM (Customer Relationship Management) system with added AI-powered features for medical professionals. It seems to leverage a Next.js frontend and a Python backend.

**Key Areas & Functionality:**

*   [x] **Frontend Framework:** Next.js (`.next/`, `app/`, `next.config.js`, `tsconfig.json`, `package.json`)
    *   [x] **UI Components:** Reusable React components (`components/`) including navigation (`navbar.tsx`), potentially a footer (`footer.tsx`), icons (`icons.tsx`), theme switching (`theme-switch.tsx`), and specialized components for voice interaction (`components/voice/`), differential diagnosis (`components/differential-diagnosis/`), and scheduling (`components/scheduler/`).
    *   [x] **Application Pages/Routing:** Defined within the `app/` directory. Key features seem to include:
        *   Voice Recording / Interaction (`app/voice-recorder-test/`, `app/voice-demo/`)
        *   Data Extraction (`app/data-extractor/`)
        *   Differential Diagnosis (`app/differential-diagnosis/`)
        *   Appointment Scheduling (`app/appointment-scheduler/`)
        *   Appointment Management (`app/appointment-manager/`)
    *   [x] **State Management & Logic:** Custom React Hooks (`hooks/`) are used, particularly for handling audio capture (`useAudioCapture.ts`), processing (`useAudioProcessing.ts`), and transcription (`useTranscription.ts`).
    *   [x] **Styling:** Tailwind CSS is used for styling (`tailwind.config.js`, `postcss.config.js`, `styles/`).
*   [x] **Backend Services:** Python-based (`backend/`)
    *   [x] **API:** Likely a primary API definition (`backend/api.py`) possibly using a framework like FastAPI or Flask.
    *   [x] **Database:** An SQLite database (`backend/healthcare.db`) is used for data persistence, with utility functions for interaction (`backend/database_utils.py`, `backend/database.py`) and setup (`backend/database_setup.py`).
    *   [x] **AI/ML Features:**
        *   Speech-to-Text (STT): Client (`backend/STT_client.py`) and potentially server (`backend/STT_server.py`) components.
        *   Text-to-Speech (TTS): Client (`backend/XTTS_client.py`) and adapter (`backend/XTTS_adapter.py`).
        *   Unified Speech Service: Integration point for STT/TTS (`backend/unified_ai_service.py`, `.env` likely contains `SPEECH_SERVICE_URL`).
        *   Medical Assistance: Core logic potentially involving diagnosis or assistance (`backend/medical_assistant.py`, `backend/differential_diagnoses.py`).
        *   Prompting: Utilizes predefined prompts (`backend/prompt.json`).
    *   [x] **Scheduling:** Logic for appointment scheduling (`backend/scheduler.py`, `backend/schedulling.py`) and management (`backend/appointment_manager.py`, `backend/appointment_manager_api.py`).
    *   [x] **Dependencies:** Python dependencies are managed via `backend/requirements.txt`.
*   [x] **Service Layer:** Communication layer between frontend and backend/external services (`services/`), specifically for audio (`services/audio/`) and data extraction (`services/extraction/`).
*   [x] **Configuration:** Environment variables (`.env`, `.env.local`) are used for configuration, including API keys and service URLs.
*   [x] **Version Control:** Git (`.git/`, `.gitignore`) is used for version control.
*   [ ] **Testing:** Some test files exist (`backend/test_diagnosis_websocket.py`, `backend/test_unified_service.py`), but the explicit instruction was not to create *new* test files. The current state indicates some testing infrastructure is present.
- [x] **Logging:** Log files (`ai_service.log`, `api.log`, `appointment_scheduler.log`, `speech_assistant.log`) suggest logging is implemented for various services. 


TODOS:
cleanup repo