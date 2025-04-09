/**
 * DataExtractionService
 * Handles extracting structured data from transcriptions via API calls
 */

import { ExtractedDataResponse } from '../../app/api/types';

export interface ExtractionOptions {
  apiUrl?: string;
  includeSOAPNote?: boolean;
}

class DataExtractionService {
  private apiBaseUrl: string;

  constructor(options: ExtractionOptions = {}) {
    this.apiBaseUrl = options.apiUrl || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  }

  /**
   * Extract data from a text transcription
   */
  async extractFromText(transcript: string): Promise<ExtractedDataResponse> {
    if (!transcript || transcript.trim() === '') {
      throw new Error('No transcription provided for extraction');
    }
    
    // Reject "Thank you" responses which are often error indicators
    if (transcript.trim() === 'Thank you.' || transcript.trim() === 'Thank you') {
      throw new Error('Invalid transcription: Default response detected');
    }
    
    console.log('Sending transcript to extraction API:', transcript.substring(0, 100) + '...');
    
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/extract/data`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ transcript }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(
          errorData?.detail || `HTTP error! status: ${response.status}`
        );
      }

      const responseData = await response.json();
      
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

  /**
   * Extract data directly from audio, handles both transcription and extraction
   */
  async extractFromAudio(audioData: string, sessionId?: string): Promise<ExtractedDataResponse> {
    console.log('Processing audio for transcription and extraction');
    
    try {
      // First get the transcription
      const transcription = await this.transcribeAudio(audioData, sessionId);
      
      if (!transcription) {
        throw new Error('Failed to transcribe audio');
      }
      
      // Then extract data from the transcription
      return await this.extractFromText(transcription);
    } catch (error) {
      console.error('Error processing audio:', error);
      throw error instanceof Error 
        ? error 
        : new Error('Failed to extract data from audio');
    }
  }

  /**
   * Transcribe audio using the speech service
   */
  private async transcribeAudio(audioData: string, sessionId?: string): Promise<string> {
    try {
      // Use the speech service URL from environment if available
      const baseUrl = process.env.NEXT_PUBLIC_SPEECH_SERVICE_URL?.replace(/\/+$/, '') || 
                      process.env.NEXT_PUBLIC_STT_SERVER_URL?.replace(/\/+$/, '') || 
                      this.apiBaseUrl;
      
      const transcribeUrl = `${baseUrl}/transcribe`;
      console.log("Using transcription URL:", transcribeUrl);
      
      // Send to STT service
      const response = await fetch(transcribeUrl, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ 
          audio: audioData,
          session_id: sessionId || false,
          maintain_context: false // Disable context to avoid errors
        })
      });

      if (!response.ok) {
        throw new Error(`Transcription failed with status: ${response.status}`);
      }

      const result = await response.json();
      
      if (!result.transcription) {
        throw new Error('No transcription received from server');
      }
      
      // Reject "Thank you" responses which are often error indicators
      if (result.transcription.trim() === 'Thank you.' || 
          result.transcription.trim() === 'Thank you') {
        throw new Error('Speech recognition error: Default response detected');
      }

      return result.transcription;
    } catch (error) {
      console.error('Error transcribing audio:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const dataExtractionService = new DataExtractionService();

// Also export the class for custom instances
export default DataExtractionService; 