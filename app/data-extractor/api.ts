const API_BASE_URL = process.env.NEXT_PUBLIC_STT_SERVER_URL|| 'http://localhost:8000';

export interface TranscriptionStatus {
  isRecording: boolean;
  duration: number;
  status: "idle" | "recording" | "processing" | "done" | "error";
}

export interface SOAPNote {
  SOAP: {
    Subjective: {
      ChiefComplaint: string;
      HistoryOfPresentIllness: {
        Onset: string;
        Location: string;
        Duration: string;
        Characteristics: string;
        AggravatingFactors: string;
        RelievingFactors: string;
        Timing: string;
        Severity: string;
      };
      PastMedicalHistory: string;
      FamilyHistory: string;
      SocialHistory: string;
      ReviewOfSystems: string;
    };
    Assessment: {
      PrimaryDiagnosis: string;
      DifferentialDiagnosis: string;
      ProblemList: string;
    };
    Plan: {
      TreatmentAndMedications: string;
      FurtherTestingOrImaging: string;
      PatientEducation: string;
      FollowUp: string;
    };
  };
}

export interface ExtractedDataResponse {
  name: string | null;
  dob: string | null;
  address: string | null;
  phone: string | null;
  email: string | null;
  insurance: string | null;
  condition: string | null;
  symptoms: string[] | null;
  reason_for_visit: string | null;
  appointment_details: {
    type: string | null;
    time: string | null;
    doctor: string | null;
    scheduled_date: string | null;
  } | null;
  metadata?: {
    timestamp: string;
    processed_at: string;
    version: string;
  };
  patient_id?: number;
  files?: {
    raw_yaml: string;
    processed_yaml: string;
    soap_note?: string;
  };
  medications?: Array<{
    name: string;
    dosage?: string;
    frequency?: string;
  }>;
  allergies?: string[];
  visit_reason?: string;
  visit_date?: string;
  doctor?: string;
  notes?: string;
  transcription?: string;
  raw_yaml?: string;
  processed_yaml?: string;
  status?: string;
  message?: string;
  soap_note?: SOAPNote;
}

export interface PatientDetails {
  patient_id: number;
  id?: number;
  name: string;
  phone: string;
  email: string | null;
  address: string | null;
  date_of_birth: string | null;
  dob?: string | null;
  insurance?: string | null;
}

export interface AppointmentDetails {
  appointment_id?: number;
  patient_id?: number;
  appointment_type?: string;
  scheduled_date?: string;
  scheduled_time?: string;
  doctor_name?: string;
  status?: string;
  notes?: string;
  date?: string;
  datetime?: string;
  doctor?: string;
}

export interface SymptomDetails {
  symptom_id: number;
  name: string;
  severity: string | null;
}

export interface MedicalConditionDetails {
  condition_id: number;
  name: string;
  date_recorded: string | null;
  notes: string | null;
  symptoms: SymptomDetails[];
}

export interface PatientDetailsResponse {
  status: string;
  patient: PatientDetails;
  appointments: AppointmentDetails[];
  medical_conditions: MedicalConditionDetails[];
  visits?: AppointmentDetails[];
  medical_history?: string;
  medications?: Array<{
    name: string;
    dosage?: string;
    frequency?: string;
  }>;
  allergies?: string[];
}

export interface PatientsListResponse {
  status: string;
  patients: PatientDetails[];
}

export async function extractData(transcript: string): Promise<ExtractedDataResponse> {
  if (!transcript || transcript.trim() === '') {
    throw new Error('No transcription provided for extraction');
  }
  
  // Explicitly reject "Thank you" responses
  if (transcript.trim() === 'Thank you.' || transcript.trim() === 'Thank you') {
    throw new Error('Invalid transcription: Default response detected');
  }
  
  console.log('Sending transcript to extraction API:', transcript.substring(0, 100) + '...');
  
  try {
    const response = await fetch(`${API_BASE_URL}/api/extract/data`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ transcript }),
    });

    console.log('API Response status:', response.status);

    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      console.error('API Error:', errorData);
      throw new Error(
        errorData?.detail || `HTTP error! status: ${response.status}`
      );
    }

    const responseData = await response.json();
    console.log('Extracted data:', responseData);
    
    // Return the complete response including YAML content
    return {
      ...responseData.data,
      status: responseData.status,
      message: responseData.message,
      raw_yaml: responseData.raw_yaml,
      processed_yaml: responseData.processed_yaml,
      files: responseData.files,
      soap_note: responseData.soap_note,
      transcription: transcript
    };
  } catch (error) {
    console.error('Error extracting data:', error);
    throw error instanceof Error 
      ? error 
      : new Error('Failed to extract data from transcript');
  }
}

export async function extractFromAudio(audioData: string, sessionId?: string): Promise<ExtractedDataResponse> {
  console.log('Sending audio for processing, session ID:', sessionId || 'none');
  
  try {
    // Use the same URL structure as in SchedulerConversation.tsx
    const baseUrl = process.env.NEXT_PUBLIC_SPEECH_SERVICE_URL?.replace(/\/+$/, '') || API_BASE_URL;
    const transcribeUrl = `${baseUrl}/transcribe`;
    console.log("Transcription URL:", transcribeUrl);
    
    // Send to STT service
    const response = await fetch(transcribeUrl, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ 
        audio: audioData,
        session_id: sessionId || false
      })
    });

    if (!response.ok) {
      console.error('Transcription API Error:', response.status);
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    
    if (!result.transcription) {
      throw new Error('No transcription received from server');
    }
    
    // Check for the common error case - reject "Thank you." transcriptions
    if (result.transcription.trim() === 'Thank you.' || 
        result.transcription.trim() === 'Thank you') {
      console.error('Default error transcription received:', result.transcription);
      throw new Error('Speech recognition failed: Default response detected');
    }

    console.log('Transcription result:', result.transcription);
    
    // Now extract the data from the transcription
    return await extractData(result.transcription);
  } catch (error) {
    console.error('Error extracting data from audio:', error);
    throw error instanceof Error 
      ? error 
      : new Error('Failed to extract data from audio');
  }
}

export async function getPatientDetails(patientId: number): Promise<PatientDetailsResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/patient/${patientId}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      console.error('API Error:', errorData);
      throw new Error(
        errorData?.detail || `HTTP error! status: ${response.status}`
      );
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error getting patient details:', error);
    throw error instanceof Error 
      ? error 
      : new Error('Failed to get patient details');
  }
}

export async function getAllPatients(): Promise<PatientsListResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/patients`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      console.error('API Error:', errorData);
      throw new Error(
        errorData?.detail || `HTTP error! status: ${response.status}`
      );
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error getting patients list:', error);
    throw error instanceof Error 
      ? error 
      : new Error('Failed to get patients list');
  }
}