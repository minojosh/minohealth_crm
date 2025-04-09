'use client';

import React, { useState } from 'react';
import VoiceRecorder from '../../components/voice/VoiceRecorderRefactored';
import { TranscriptionStatus } from '../api/types';

const VoiceRecorderTestPage: React.FC = () => {
  const [transcription, setTranscription] = useState<string>('');
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [status, setStatus] = useState<TranscriptionStatus>({
    isRecording: false,
    duration: 0,
    status: 'idle',
  });

  // Handle transcription updates
  const handleTranscriptionUpdate = (text: string, id?: string) => {
    setTranscription(text);
    if (id) setSessionId(id);
  };

  // Handle status changes
  const handleStatusChange = (newStatus: TranscriptionStatus) => {
    setStatus(newStatus);
  };

  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <h1 className="text-3xl font-bold text-center mb-8">
          Voice Recorder Test Page
        </h1>
        
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Record Audio</h2>
          
          <div className="flex flex-col items-center">
            <VoiceRecorder
              onTranscriptionUpdate={handleTranscriptionUpdate}
              onRecordingStatusChange={handleStatusChange}
              showAudioVisualizer={true}
              useHighQualityProcessing={true}
            />
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Status</h2>
          
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-gray-50 p-3 rounded">
              <p className="text-sm font-medium text-gray-500">Recording Status</p>
              <p className="font-mono">{status.status}</p>
            </div>
            
            <div className="bg-gray-50 p-3 rounded">
              <p className="text-sm font-medium text-gray-500">Is Recording</p>
              <p className="font-mono">{status.isRecording ? 'Yes' : 'No'}</p>
            </div>
            
            <div className="bg-gray-50 p-3 rounded">
              <p className="text-sm font-medium text-gray-500">Duration</p>
              <p className="font-mono">{status.duration.toFixed(1)}s</p>
            </div>
            
            <div className="bg-gray-50 p-3 rounded">
              <p className="text-sm font-medium text-gray-500">Session ID</p>
              <p className="font-mono text-xs overflow-hidden text-ellipsis">{sessionId || 'None'}</p>
            </div>
          </div>
        </div>
        
        {transcription && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">Transcription Result</h2>
            <div className="bg-gray-50 p-4 rounded">
              <p className="whitespace-pre-wrap">{transcription}</p>
            </div>
          </div>
        )}
        
        <div className="mt-8 text-center text-sm text-gray-500">
          <p>
            This is a test page for the refactored Voice Recorder component with improved audio handling.
          </p>
        </div>
      </div>
    </div>
  );
};

export default VoiceRecorderTestPage; 