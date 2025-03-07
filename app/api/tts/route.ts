import { NextRequest, NextResponse } from 'next/server';
import { TextToSpeechRequest, TextToSpeechResponse } from '../types';

export async function POST(request: NextRequest) {
  try {
    const body: TextToSpeechRequest = await request.json();
    
    // Validate request
    if (!body.text) {
      return NextResponse.json(
        { error: 'Text is required' },
        { status: 400 }
      );
    }

    // Call the backend TTS service
    const backendUrl = process.env.BACKEND_API_URL || 'http://localhost:8000';
    const response = await fetch(`${backendUrl}/generate_speech`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: body.text,
        speaker: body.speaker || 'emma', // Default speaker
        speed: body.speed || 1.0,       // Default speed
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json(
        { error: errorData.message || 'Failed to synthesize speech' },
        { status: response.status }
      );
    }

    const data = await response.json();
    
    const result: TextToSpeechResponse = {
      audioData: data.audio_base64,
      duration: data.duration,
    };

    return NextResponse.json(result);
  } catch (error) {
    console.error('TTS API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
