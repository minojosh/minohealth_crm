import { useState, useEffect, useRef } from 'react';
import { Button } from "@heroui/button";
import { audioService } from '../../app/api/audio';
import { TranscriptionStatus } from '../../app/api/types';

export default function CombinedVoiceRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [status, setStatus] = useState({
    isRecording: false,
    duration: 0,
    status: 'idle',
  });
  const [error, setError] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioVisualizerRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);

  useEffect(() => {
    audioService.connect();

    audioService.onTranscription((response) => {
      if (response.error) {
        setError(response.error);
        return;
      }
      
      const transcriptionText = response.text || response.transcription || '';
      
      if (transcriptionText) {
        setTranscription(transcriptionText);
        
        if (response.isComplete) {
          setIsProcessing(false);
          setStatus({
            isRecording: false,
            duration: status.duration,
            status: 'done',
          });
        }
      }
    });

    audioService.onError((errorMsg) => {
      setError(errorMsg);
      setIsProcessing(false);
      stopRecording();
    });

    return () => {
      stopVisualization();
      if (mediaRecorderRef.current && isRecording) {
        mediaRecorderRef.current.stop();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  const startRecording = async () => {
    try {
      setError(null);
      setTranscription('');
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000
        } 
      });
      
      startVisualization(stream);
      
      const recorder = await audioService.startTranscription(stream);
      mediaRecorderRef.current = recorder;
      
      setIsRecording(true);
      setStatus({
        isRecording: true,
        duration: 0,
        status: 'recording',
      });
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to start recording');
      stopVisualization();
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
      
      setIsRecording(false);
      setIsProcessing(true);
      setStatus({
        isRecording: false,
        duration: 0,
        status: 'processing',
      });
    }
    
    stopVisualization();
  };

  const startVisualization = (stream) => {
    if (!audioVisualizerRef.current) return;
    
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioCtx;
      
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.8;
      analyserRef.current = analyser;
      
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);
      
      drawVisualization();
    } catch (error) {
      console.error('Error starting visualization:', error);
    }
  };

  const drawVisualization = () => {
    if (!audioVisualizerRef.current || !analyserRef.current) return;
    
    const canvas = audioVisualizerRef.current;
    const canvasCtx = canvas.getContext('2d');
    if (!canvasCtx) return;
    
    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      if (!isRecording) return;
      animationFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      
      const barWidth = (canvas.width / bufferLength) * 2.5;
      let x = 0;
      
      const gradient = canvasCtx.createLinearGradient(0, 0, 0, canvas.height);
      gradient.addColorStop(0, '#4f46e5');
      gradient.addColorStop(1, '#818cf8');
      
      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height;
        canvasCtx.fillStyle = gradient;
        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        x += barWidth + 1;
      }
    };
    
    draw();
  };

  const stopVisualization = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(err => {
        console.warn('Error closing audio context:', err);
      });
      audioContextRef.current = null;
    }
    
    if (audioVisualizerRef.current) {
      const canvasCtx = audioVisualizerRef.current.getContext('2d');
      if (canvasCtx) {
        canvasCtx.clearRect(0, 0, audioVisualizerRef.current.width, audioVisualizerRef.current.height);
      }
    }
  };

  const resetRecorder = () => {
    setTranscription('');
    setStatus({
      isRecording: false,
      duration: 0,
      status: 'idle',
    });
  };

  return (
    <div className="flex flex-col gap-4 max-w-2xl mx-auto">
      <div className="flex items-center justify-center h-64 bg-gradient-to-b from-gray-900 to-gray-800 rounded-xl shadow-lg overflow-hidden relative">
        {isRecording ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-4">
            <div className="absolute top-3 left-3 flex items-center">
              <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse mr-2"></div>
              <span className="text-white text-xs font-medium">Recording</span>
            </div>
            <canvas 
              ref={audioVisualizerRef} 
              className="w-full h-full"
              width={400}
              height={200}
            />
          </div>
        ) : isProcessing ? (
          <motion.div 
            className="text-center p-6"
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="w-12 h-12 border-4 border-t-indigo-500 border-r-transparent border-b-indigo-500 border-l-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-white font-medium">Processing your audio...</p>
            <p className="text-gray-400 text-sm mt-2">This may take a few moments</p>
          </motion.div>
        ) : transcription ? (
          <div className="p-6 w-full h-full flex flex-col">
            <div className="flex-1 overflow-auto bg-gray-900/50 p-4 rounded-md text-gray-200 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-gray-900">
              {transcription}
            </div>
            <div className="flex justify-end mt-4">
              <Button 
                color="primary" 
                size="sm"
                onClick={resetRecorder}
                className="text-sm px-3 py-1"
                startContent={
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                  </svg>
                }
              >
                New Recording
              </Button>
            </div>
          </div>
        ) : (
          <div className="text-center p-6">
            <div className="w-16 h-16 bg-gray-700 rounded-full flex items-center justify-center mb-4 mx-auto">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 15C13.66 15 15 13.66 15 12V6C15 4.34 13.66 3 12 3C10.34 3 9 4.34 9 6V12C9 13.66 10.34 15 12 15Z" fill="#4f46e5"/>
                <path d="M17 12C17 14.76 14.76 17 12 17C9.24 17 7 14.76 7 12H5C5 15.53 7.61 18.43 11 18.92V21H13V18.92C16.39 18.43 19 15.53 19 12H17Z" fill="#4f46e5"/>
              </svg>
            </div>
            <p className="text-white font-medium text-lg">Ready to Record</p>
            <p className="text-gray-400 text-sm mt-2">Click the button below to start recording</p>
          </div>
        )}
      </div>
      
      {!transcription && (
        <div className="flex justify-center gap-3 mt-2">
          {!isRecording ? (
            <Button 
              color="primary" 
              size="lg"
              onClick={startRecording}
              isDisabled={isProcessing}
              className="px-6 py-2 rounded-full shadow-lg hover:shadow-xl transition-all duration-300"
              startContent={
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 15C13.66 15 15 13.66 15 12V6C15 4.34 13.66 3 12 3C10.34 3 9 4.34 9 6V12C9 13.66 10.34 15 12 15Z" fill="currentColor"/>
                  <path d="M17 12C17 14.76 14.76 17 12 17C9.24 17 7 14.76 7 12H5C5 15.53 7.61 18.43 11 18.92V21H13V18.92C16.39 18.43 19 15.53 19 12H17Z" fill="currentColor"/>
                </svg>
              }
            >
              Start Recording
            </Button>
          ) : (
            <Button 
              color="danger" 
              size="lg"
              onClick={stopRecording}
              className="px-6 py-2 rounded-full shadow-lg hover:shadow-xl transition-all duration-300"
              startContent={
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M6 6H18V18H6V6Z" fill="currentColor"/>
                </svg>
              }
            >
              Stop Recording
            </Button>
          )}
        </div>
      )}
      
      {status.status !== 'idle' && !transcription && (
        <div className="bg-gray-800 rounded-lg p-3 mt-2">
          <div className="flex items-center justify-between">
            <span className="text-gray-300 text-sm">Status:</span>
            <span className={`text-sm font-medium px-2 py-1 rounded-full text-white
              ${status.status === 'recording' ? 'bg-red-500' : 
                status.status === 'processing' ? 'bg-yellow-500' : 
                status.status === 'done' ? 'bg-green-500' : 'bg-gray-500'}`}
            >
              {status.status === 'recording' ? 'Recording' : 
               status.status === 'processing' ? 'Processing' : 
               status.status === 'done' ? 'Completed' : 'Idle'}
            </span>
          </div>
        </div>
      )}
      
      {error && (
        <div className="bg-red-900/30 border border-red-500 text-red-300 text-sm p-3 rounded-lg mt-2 flex items-start">
          <svg className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}