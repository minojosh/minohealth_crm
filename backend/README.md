# minoHealth CRM Backend

This is the backend service for the minoHealth CRM system, providing API endpoints for speech recognition, text-to-speech, scheduling, differential diagnosis, and medical data extraction.

## Recent Updates

The API has been unified and improved with the following changes:

1. **Unified AI Service Architecture**: 
   - Created a unified service layer in `unified_ai_service.py`
   - Modular design with specialized services for different functionalities
   - Better code organization and reusability

2. **Medical Assistant Integration**:
   - Added `MedicalAssistantService` with data extraction capabilities
   - Improved SOAP note generation with proper formatting
   - Enhanced patient database integration

3. **API Improvements**:
   - Added streaming TTS support via `/tts-stream` endpoint
   - Maintained backward compatibility with existing frontend code
   - Streamlined and documented API endpoints

4. **Deprecated Endpoints**:
   - Marked outdated endpoints with `deprecated=True`
   - Provided compatibility layers for smooth transition

## API Endpoints

### Active Endpoints

1. `/tts` - Text-to-speech conversion (base64 audio)
2. `/tts-stream` - Streaming text-to-speech (chunked audio)
3. `/transcribe` - Speech-to-text transcription
4. `/ws/audio` - WebSocket for real-time audio streaming
5. `/ws/schedule/{patient_id}` - WebSocket for appointment scheduling
6. `/ws/conversation` - WebSocket for medication and appointment conversations
7. `/api/medical/extract` - Extract structured medical data from transcripts
8. `/api/medical/soap` - Generate SOAP notes from conversations
9. `/health` - Health check endpoint

### Deprecated Endpoints (to be removed)

1. `/api/extract/data` - Use `/api/medical/extract` instead

## Setup and Installation

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Update the values as needed

3. Initialize the database:
   ```
   python database_setup.py
   ```

4. Run the server:
   ```
   uvicorn api:app --reload
   ```

## AI Service Modules

### Unified AI Service Architecture

The system uses a unified service architecture with the following components:

1. **Base AI Service**: Common functionality for all AI interactions
   - TTS integration
   - STT integration
   - Moremi AI integration

2. **Specialized Services**:
   - `SchedulerService`: For appointment scheduling
   - `DifferentialDiagnosisService`: For medical diagnosis
   - `MedicationReminderService`: For medication reminders
   - `MedicalAssistantService`: For data extraction and SOAP notes

## Frontend Compatibility

All endpoints are backward compatible with the existing frontend implementation. The new endpoints provide enhanced functionality while maintaining the same API contract that the frontend expects.

## Data Flow

1. **Audio Input** → STT → Transcript → AI Processing → Response → TTS → **Audio Output**
2. **Text Input** → AI Processing → Structured Data/SOAP Note → **JSON Output**

## Dependencies

- FastAPI - Web framework
- Uvicorn - ASGI server
- PyYAML - YAML processing
- NumPy - Numerical processing
- Requests - HTTP client
