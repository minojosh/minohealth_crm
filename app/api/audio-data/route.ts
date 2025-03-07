import { NextRequest, NextResponse } from 'next/server';
import { getAudioData, storeAudioData } from '../audio';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const sessionId = searchParams.get('sessionId');
  
  if (!sessionId) {
    return NextResponse.json({ error: 'Session ID is required' }, { status: 400 });
  }
  
  const audioData = getAudioData(sessionId);
  
  if (!audioData) {
    return NextResponse.json({ error: 'No audio data found for this session' }, { status: 404 });
  }
  
  return NextResponse.json({ audioData });
}

export async function POST(request: NextRequest) {
  try {
    const { sessionId, audioData } = await request.json();
    
    if (!sessionId || !audioData) {
      return NextResponse.json(
        { error: 'Session ID and audio data are required' }, 
        { status: 400 }
      );
    }
    
    storeAudioData(sessionId, audioData);
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error storing audio data:', error);
    return NextResponse.json(
      { error: 'Failed to store audio data' }, 
      { status: 500 }
    );
  }
}