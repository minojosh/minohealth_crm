const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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
  };
}

export interface PatientDetails {
  patient_id: number;
  name: string;
  phone: string;
  email: string | null;
  address: string | null;
  date_of_birth: string | null;
}

export interface AppointmentDetails {
  appointment_id: number;
  datetime: string | null;
  appointment_type: string | null;
  status: string;
  doctor_id: number | null;
  doctor_name: string | null;
  notes: string | null;
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
}

export interface PatientsListResponse {
  status: string;
  patients: PatientDetails[];
}

export async function extractData(transcript: string): Promise<ExtractedDataResponse> {
  console.log('Sending transcript to extraction API:', transcript);
  
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

    const data = await response.json();
    console.log('Extracted data:', data);
    return data.data;
  } catch (error) {
    console.error('Error extracting data:', error);
    throw error instanceof Error 
      ? error 
      : new Error('Failed to extract data from transcript');
  }
}

export async function extractFromAudio(audioBase64: string): Promise<ExtractedDataResponse> {
  console.log('Sending audio to extraction API');
  
  try {
    const response = await fetch(`${API_BASE_URL}/api/extract/audio`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ audio: audioBase64 }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      console.error('API Error:', errorData);
      throw new Error(
        errorData?.detail || `HTTP error! status: ${response.status}`
      );
    }

    const data = await response.json();
    console.log('Extracted data from audio:', data);
    return data.data;
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