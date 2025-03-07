import { NextRequest, NextResponse } from 'next/server';
import { AudioTranscriptionRequest, AudioTranscriptionResponse } from '../types';

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

    // Call the backend STT service
    const backendUrl = process.env.BACKEND_API_URL || 'http://localhost:8000';
    const response = await fetch(`${backendUrl}/transcribe`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        audio_data: body.audioData,
        language: body.language || 'en',  // Default to English
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json(
        { error: errorData.message || 'Failed to transcribe audio' },
        { status: response.status }
      );
    }

    const data = await response.json();
    
    const result: AudioTranscriptionResponse = {
      text: data.text,
      confidence: data.confidence,
      segments: data.segments?.map((segment: any) => ({
        text: segment.text,
        start: segment.start,
        end: segment.end,
        speaker: segment.speaker
      })),
    };

    return NextResponse.json(result);
  } catch (error) {
    console.error('Transcription API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
