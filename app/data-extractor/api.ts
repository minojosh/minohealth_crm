const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface ExtractedDataResponse {
  name: string | null;
  dob: string | null;
  address: string | null;
  phone: string | null;
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
    return data;
  } catch (error) {
    console.error('Error extracting data:', error);
    throw error instanceof Error 
      ? error 
      : new Error('Failed to extract data from transcript');
  }
}