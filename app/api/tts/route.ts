import { NextRequest, NextResponse } from 'next/server';
import { TextToSpeechRequest, TextToSpeechResponse } from '../types';
import { API_ENDPOINTS } from '../api';

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
    const response = await fetch(API_ENDPOINTS.tts, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: body.text,
        speaker: body.speaker || 'default',
        speed: body.speed || 1.0,
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
    
    // Create response with proper typing
    const result: TextToSpeechResponse = {
      audioData: data.audio,
      duration: data.duration || 0,
      sampleRate: data.sample_rate || 24000, // Use sample_rate from backend or default to 24000
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
