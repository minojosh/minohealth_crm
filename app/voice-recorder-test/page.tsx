"use client"
import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Play, Pause, StopCircle, Save, FileText, Clipboard, AlertTriangle, Volume } from 'lucide-react';
import VoiceRecorderRefactored from '../../components/voice/VoiceRecorderRefactored';
import { TranscriptionStatus } from '../../app/api/types';

interface VoiceRecorderTestProps {
  className?: string;
}

const VoiceRecorderTest: React.FC<VoiceRecorderTestProps> = ({ className }) => {
  // State management
  const [transcription, setTranscription] = useState<string>('');
  const [recordingStatus, setRecordingStatus] = useState<TranscriptionStatus>({
    isRecording: false,
    status: 'idle',
    duration: 0
  });
  const [copied, setCopied] = useState(false);
  const [showControls, setShowControls] = useState(true);
  const [audioLevel, setAudioLevel] = useState<number[]>(Array(20).fill(0));
  const audioLevelRef = useRef<number[]>(Array(20).fill(0));
  const animationRef = useRef<number>(0);
  
  // Format recording time to MM:SS
  const formatTime = (duration: number): string => {
    const minutes = Math.floor(duration / 60);
    const seconds = Math.floor(duration % 60);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };
  
  // Handle transcription update from voice recorder
  const handleTranscriptionUpdate = (text: string) => {
    setTranscription(text);
  };
  
  // Handle recording status change
  const handleRecordingStatusChange = (status: TranscriptionStatus) => {
    setRecordingStatus(status);
    
    // Start or stop audio level visualization
    if (status.isRecording) {
      startLevelAnimation();
    } else {
      stopLevelAnimation();
    }
  };
  
  // Animation for audio levels
  const updateAudioLevels = () => {
    if (recordingStatus.isRecording) {
      // Generate random levels with weighted previous values for smoothness
      const newLevels = audioLevelRef.current.map((level) => {
        const randomFactor = Math.random() * 0.5;
        return Math.min(1, Math.max(0.05, level * 0.6 + randomFactor));
      });
      
      audioLevelRef.current = newLevels;
      setAudioLevel(newLevels);
      animationRef.current = requestAnimationFrame(updateAudioLevels);
    }
  };
  
  const startLevelAnimation = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    animationRef.current = requestAnimationFrame(updateAudioLevels);
  };
  
  const stopLevelAnimation = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = 0;
    }
    // Reset levels to zero with small delay between each
    audioLevelRef.current.forEach((_, index) => {
      setTimeout(() => {
        const newLevels = [...audioLevelRef.current];
        newLevels[index] = 0;
        audioLevelRef.current = newLevels;
        setAudioLevel(newLevels);
      }, index * 20);
    });
  };
  
  // Clean up animation frame on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);
  
  // Copy transcription to clipboard
  const copyToClipboard = () => {
    if (!transcription) return;
    
    navigator.clipboard.writeText(transcription)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch(err => {
        console.error('Failed to copy: ', err);
      });
  };
  
  // Reset transcription
  const resetTranscription = () => {
    setTranscription('');
  };
  
  // Toggle controls visibility
  const toggleControls = () => {
    setShowControls(!showControls);
  };
  
  // Click hidden mic button to start recording
  const triggerRecording = () => {
    const micBtn = document.querySelector('[aria-label="Toggle Recording"]');
    if (micBtn instanceof HTMLButtonElement) {
      micBtn.click();
    }
  };
  
  return (
    <div className="max-w-4xl mx-auto px-4 py-6">
      <div className="bg-white rounded-md shadow-sm border border-gray-200">
        {/* Header */}
        <div className="bg-gray-50 border-b border-gray-200 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-1.5 bg-blue-100 rounded-md">
              <Volume className="h-5 w-5 text-blue-700" />
            </div>
            <h1 className="text-xl font-semibold text-gray-800">Medical Voice Recognition</h1>
          </div>
          
          <div className="flex items-center">
            <span 
              className={`px-3 py-1 rounded-full text-xs font-medium ${
                recordingStatus.isRecording 
                  ? 'bg-red-100 text-red-700' 
                  : recordingStatus.status === 'processing'
                  ? 'bg-yellow-100 text-yellow-700'
                  : 'bg-green-100 text-green-700'
              }`}
            >
              {recordingStatus.isRecording 
                ? 'Recording' 
                : recordingStatus.status === 'processing'
                ? 'Processing'
                : 'Ready'}
            </span>
          </div>
        </div>
        
        {/* Main Content */}
        <div className="px-6 py-4">
          {/* Hidden original recorder component */}
          <div className="hidden">
            <VoiceRecorderRefactored 
              onTranscriptionUpdate={handleTranscriptionUpdate}
              onRecordingStatusChange={handleRecordingStatusChange}
              showAudioVisualizer={false}
              useHighQualityProcessing={true}
            />
          </div>
          
          {/* Custom UI */}
          <div className="flex flex-col lg:flex-row gap-6">
            {/* Recorder Section */}
            <div className="flex-1 min-w-[300px]">
              {/* Audio Visualizer */}
              <div className="h-24 bg-gray-50 border border-gray-200 rounded-md flex items-end justify-between p-2 mb-4">
                {audioLevel.map((level, i) => (
                  <div 
                    key={i} 
                    className="w-[3px] bg-blue-500 rounded-t transition-all duration-100 ease-out"
                    style={{ 
                      height: `${level * 80}%`,
                      opacity: recordingStatus.isRecording ? 1 : 0.4
                    }}
                  />
                ))}
              </div>
              
              {/* Recording Time */}
              <div className="font-mono text-2xl text-center mb-4">
                {formatTime(recordingStatus.duration)}
              </div>
              
              {/* Controls */}
              <div className="flex items-center justify-center space-x-4 mb-6">
                <button
                  onClick={triggerRecording}
                  className={`
                    p-4 rounded-full transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 
                    ${recordingStatus.isRecording 
                      ? 'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500' 
                      : 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500'}
                  `}
                  aria-label="Start or stop recording"
                >
                  {recordingStatus.isRecording ? (
                    <StopCircle className="h-8 w-8" />
                  ) : (
                    <Mic className="h-8 w-8" />
                  )}
                </button>
              </div>
              
              {/* Status Text */}
              <div className="text-center text-gray-600 text-sm">
                {recordingStatus.isRecording ? (
                  <span className="text-red-600 font-medium">Recording in progress</span>
                ) : recordingStatus.status === 'processing' ? (
                  <span className="text-yellow-600 font-medium">Processing audio...</span>
                ) : transcription ? (
                  <span className="text-green-600 font-medium">Transcription complete</span>
                ) : (
                  <span>Click the microphone to start recording</span>
                )}
              </div>
              
              {/* Recording Tips */}
              <div className="mt-6 p-3 bg-blue-50 border border-blue-100 rounded text-sm text-blue-800">
                <div className="flex items-start mb-2">
                  <AlertTriangle className="h-4 w-4 mr-2 mt-0.5 flex-shrink-0" />
                  <p className="font-medium">For optimal results:</p>
                </div>
                <ul className="space-y-1 pl-6 list-disc">
                  <li>Minimize background noise</li>
                  <li>Speak at a consistent pace</li>
                  <li>Articulate clearly</li>
                  <li>Use standard medical terminology</li>
                </ul>
              </div>
            </div>
            
            {/* Transcription Section */}
            <div className="flex-1 min-w-[300px]">
              <div className="border border-gray-200 rounded-md h-full flex flex-col">
                <div className="flex items-center justify-between px-4 py-3 bg-gray-50 border-b border-gray-200">
                  <div className="flex items-center">
                    <FileText className="h-4 w-4 text-gray-500 mr-2" />
                    <h2 className="font-medium text-gray-800">Transcription</h2>
                  </div>
                  
                  <div className="flex space-x-2">
                    <button 
                      onClick={resetTranscription}
                      className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-md transition-colors"
                      aria-label="Clear transcription"
                      disabled={!transcription}
                    >
                      <StopCircle className="h-4 w-4" />
                    </button>
                    <button 
                      onClick={copyToClipboard}
                      className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-md transition-colors"
                      aria-label="Copy to clipboard"
                      disabled={!transcription}
                    >
                      <Clipboard className="h-4 w-4" />
                    </button>
                  </div>
                </div>
                
                <div className="flex-grow p-4 overflow-y-auto bg-white min-h-[280px]">
                  {transcription ? (
                    <p className="text-gray-800 whitespace-pre-wrap">{transcription}</p>
                  ) : (
                    <div className="h-full flex items-center justify-center text-gray-400 italic">
                      No transcription available
                    </div>
                  )}
                </div>
                
                <div className="px-4 py-2 bg-gray-50 border-t border-gray-200 flex justify-between items-center text-xs text-gray-500">
                  <span>{transcription ? `${transcription.length} characters` : 'Ready to record'}</span>
                  {copied && <span className="text-green-600">Copied to clipboard</span>}
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Footer */}
        <div className="px-6 py-3 border-t border-gray-200 bg-gray-50 flex justify-between text-xs text-gray-500">
          <span>MinoHealth Voice Recognition System</span>
          <span>v1.2.0</span>
        </div>
      </div>
    </div>
  );
};

export default VoiceRecorderTest;