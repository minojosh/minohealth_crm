import { NextRequest, NextResponse } from 'next/server';
import { AudioTranscriptionRequest, AudioTranscriptionResponse } from '../types';

// Suspicious response constant
const SUSPICIOUS_TRANSCRIPTION = "Thanks for watching.";

export async function POST(request: NextRequest) {
  try {
    const body: AudioTranscriptionRequest = await request.json();
    
    // Validate request
    if (!body.audioData) {
      return NextResponse.json(
        { error: 'Audio data is required' },
        { status: 400 }
      );
    }

    // Generate a new session ID if not provided
    const sessionId = body.sessionId || crypto.randomUUID();
    
    // Check if we should use an alternative method (fallback after detecting suspicious response)
    if (body.alternativeMethod) {
      console.log("Using alternative transcription method to bypass server issue");
      
      // Try a different approach - modify request parameters
      // Strategy 1: Send smaller chunks with a delay
      const result = await tryAlternativeTranscription(body.audioData, sessionId);
      
      if (result) {
        return NextResponse.json({
          text: result.transcription,
          confidence: result.confidence || 0.8,
          sessionId: sessionId,
          isComplete: true,
          alternativeUsed: true
        });
      }
    }
    
    // Call the backend STT service with standard approach
    const backendUrl = process.env.NEXT_PUBLIC_SPEECH_SERVER_URL || 'http://localhost:8000';
    const response = await fetch(`${backendUrl}/transcribe`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        audio: body.audioData,
        session_id: sessionId,
        maintain_context: body.maintainContext !== false, // Default to true
        language: body.language || 'en',  // Default to English
        process_complete: body.processComplete || false
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: errorData.message || 'Failed to transcribe audio' },
        { status: response.status }
      );
    }

    const data = await response.json();
    
    // Check if we got the suspicious response
    if (data.transcription === SUSPICIOUS_TRANSCRIPTION) {
      console.warn("Received suspicious 'Thanks for watching' transcription from server");
      
      // Try the alternative method directly in case that helps
      const alternativeResult = await tryAlternativeTranscription(body.audioData, sessionId);
      
      if (alternativeResult && alternativeResult.transcription !== SUSPICIOUS_TRANSCRIPTION) {
        return NextResponse.json({
          text: alternativeResult.transcription,
          confidence: alternativeResult.confidence || 0.8,
          sessionId: sessionId,
          isComplete: body.processComplete,
          alternativeUsed: true
        });
      }
    }
    
    // Return the original response if alternative method didn't help
    return NextResponse.json({
      text: data.transcription,
      confidence: data.confidence || 0.9,
      sessionId: data.session_id || sessionId,
      isComplete: body.processComplete
    });
  } catch (error) {
    console.error('Transcription API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

/**
 * Tries alternative approaches to transcribe audio when standard method returns suspicious results
 */
async function tryAlternativeTranscription(audioData: string, sessionId: string) {
  try {
    // Strategy 1: Try a direct request with modified parameters
    const backendUrl = process.env.NEXT_PUBLIC_SPEECH_SERVER_URL || 'http://localhost:8000';
    
    // Add a timestamp to force a new request rather than using cached results
    const timestamp = Date.now();
    
    const response = await fetch(`${backendUrl}/transcribe?t=${timestamp}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
      },
      body: JSON.stringify({
        audio: audioData,
        session_id: `alt_${sessionId}`,
        maintain_context: true,
        language: 'en',
        process_complete: true,
        bypass_cache: true,
        sample_rate: 16000  // Try explicitly setting sample rate
      }),
    });

    if (response.ok) {
      const data = await response.json();
      
      // If we got something other than the suspicious text, return it
      if (data.transcription && data.transcription !== SUSPICIOUS_TRANSCRIPTION) {
        console.log("Alternative transcription successful");
        return data;
      }
    }
    
    // Strategy 2: Try a second backend endpoint if configured
    const alternativeBackendUrl = process.env.ALT_NEXT_PUBLIC_SPEECH_SERVER_URL;
    if (alternativeBackendUrl) {
      const altResponse = await fetch(`${alternativeBackendUrl}/transcribe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          audio: audioData,
          session_id: sessionId
        }),
      });
      
      if (altResponse.ok) {
        const altData = await altResponse.json();
        if (altData.transcription && altData.transcription !== SUSPICIOUS_TRANSCRIPTION) {
          return altData;
        }
      }
    }
    
    return null; // Return null if all strategies fail
  } catch (error) {
    console.error('Alternative transcription error:', error);
    return null;
  }
}
