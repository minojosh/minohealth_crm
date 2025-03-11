import { NextRequest, NextResponse } from 'next/server';
import { API_ENDPOINTS } from '../../api';

// This API route provides the initial greeting for medication or appointment reminders
// with TTS audio generation for immediate playback
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const type = searchParams.get('type');
  const patientId = searchParams.get('patientId') || '1';

  try {
    // Default greeting based on type
    let greeting = '';
    let audioData = null;
    
    if (type === 'medication') {
      greeting = "Hello, this is Mino Healthcare following up about your medication. Have you been taking your medication as prescribed?";
    } else if (type === 'appointment') {
      greeting = "Hello, this is Mino Healthcare calling to remind you about your upcoming appointment.";
    } else {
      greeting = "Hello, this is Mino Healthcare assistant. How can I help you today?";
    }

    // Try to get TTS audio from the backend
    try {
      console.log("Fetching TTS audio for greeting...");
      const ttsResponse = await fetch(API_ENDPOINTS.tts, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: greeting,
        }),
        signal: AbortSignal.timeout(10000)
      });
      
      if (ttsResponse.ok) {
        const ttsData = await ttsResponse.json();
        if (ttsData.audio) {
          audioData = ttsData.audio;
          console.log("TTS audio received successfully");
        } else {
          console.warn("TTS response didn't contain audio data");
        }
      } else {
        console.warn(`TTS service returned error: ${ttsResponse.status} ${ttsResponse.statusText}`);
      }
    } catch (ttsError) {
      console.warn('TTS service unavailable, continuing without audio:', ttsError);
      // Continue without audio if TTS fails
    }

    console.log(`Returning greeting for ${type} conversation, audioData available: ${audioData !== null}`);
    
    // Return the greeting with audio data if available
    return NextResponse.json({
      greeting,
      audioData,
    });
  } catch (error) {
    console.error('Error generating initial greeting:', error);
    return NextResponse.json(
      { 
        error: 'Failed to fetch greeting',
        greeting: "Hello, I'm your healthcare assistant. How may I help you today?"
      },
      { status: 500 }
    );
  }
}
