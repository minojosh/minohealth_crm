import React, { useState } from 'react';
import VoiceRecorder from './VoiceRecorderRefactored';
import { TranscriptionStatus } from '../../app/api/types';
import { dataExtractionService } from '../../services/extraction/DataExtractionService';

/**
 * Example component showing how to use the refactored VoiceRecorder
 * with data extraction.
 */
const VoiceRecorderExample: React.FC = () => {
  const [transcription, setTranscription] = useState('');
  const [extractedData, setExtractedData] = useState<any>(null);
  const [status, setStatus] = useState<TranscriptionStatus>({
    isRecording: false,
    duration: 0,
    status: 'idle',
  });
  const [isExtracting, setIsExtracting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Handle transcription updates
  const handleTranscriptionUpdate = async (text: string, sessionId?: string) => {
    setTranscription(text);
    
    // If we have a complete transcription, extract data
    if (text && status.status === 'done') {
      await extractData(text);
    }
  };
  
  // Handle status changes from the recorder
  const handleStatusChange = (newStatus: TranscriptionStatus) => {
    setStatus(newStatus);
    
    // If recording has completed and we have transcription, extract data
    if (newStatus.status === 'done' && transcription) {
      extractData(transcription);
    }
  };
  
  // Extract data from transcription
  const extractData = async (text: string) => {
    if (!text.trim()) {
      setError('No transcription available to extract data from');
      return;
    }
    
    setIsExtracting(true);
    setError(null);
    
    try {
      // Use the service to extract data
      const data = await dataExtractionService.extractFromText(text);
      setExtractedData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to extract data');
    } finally {
      setIsExtracting(false);
    }
  };
  
  // Render the extracted data
  const renderExtractedData = () => {
    if (!extractedData) return null;
    
    return (
      <div className="mt-6 p-4 bg-gray-100 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">Extracted Data:</h3>
        
        {extractedData.name && (
          <div className="mb-2">
            <strong>Name:</strong> {extractedData.name}
          </div>
        )}
        
        {extractedData.condition && (
          <div className="mb-2">
            <strong>Condition:</strong> {extractedData.condition}
          </div>
        )}
        
        {extractedData.symptoms && extractedData.symptoms.length > 0 && (
          <div className="mb-2">
            <strong>Symptoms:</strong>
            <ul className="list-disc ml-6">
              {extractedData.symptoms.map((symptom: string, index: number) => (
                <li key={index}>{symptom}</li>
              ))}
            </ul>
          </div>
        )}
        
        {extractedData.reason_for_visit && (
          <div className="mb-2">
            <strong>Reason for Visit:</strong> {extractedData.reason_for_visit}
          </div>
        )}
        
        {/* View more button for full details */}
        <button 
          className="mt-2 px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
          onClick={() => alert(JSON.stringify(extractedData, null, 2))}
        >
          View All Data
        </button>
      </div>
    );
  };
  
  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Voice Data Extraction Demo</h2>
      
      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg">
          {error}
        </div>
      )}
      
      <VoiceRecorder 
        onTranscriptionUpdate={handleTranscriptionUpdate}
        onRecordingStatusChange={handleStatusChange}
        showAudioVisualizer={true}
        useHighQualityProcessing={true}
      />
      
      {transcription && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Transcription:</h3>
          <div className="p-3 bg-gray-100 rounded">
            {transcription}
          </div>
          
          {status.status === 'done' && (
            <button 
              className="mt-2 px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50"
              onClick={() => extractData(transcription)}
              disabled={isExtracting}
            >
              {isExtracting ? 'Extracting...' : 'Extract Data Again'}
            </button>
          )}
        </div>
      )}
      
      {extractedData && renderExtractedData()}
    </div>
  );
};

export default VoiceRecorderExample;