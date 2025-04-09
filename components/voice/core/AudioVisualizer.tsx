import React, { useRef, useEffect, useState } from 'react';
import { audioService } from '../../../services/audio/AudioService';

interface AudioVisualizerProps {
  audioContext: AudioContext | null;
  stream: MediaStream | null;
  isRecording: boolean;
  width?: number;
  height?: number;
  barColor?: string | string[];
  backgroundColor?: string;
  barCount?: number;
  smoothingTimeConstant?: number;
}

const AudioVisualizer: React.FC<AudioVisualizerProps> = ({
  audioContext,
  stream,
  isRecording,
  width = 300,
  height = 60,
  barColor = ['#4f46e5', '#818cf8'],
  backgroundColor = 'transparent',
  barCount = 32,
  smoothingTimeConstant = 0.8
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  
  useEffect(() => {
    // Always try to use audioService's context if available
    const context = audioService.audioContext || audioContext;
    const activeStream = stream || (audioService.currentStream as MediaStream);
    
    // Start visualization when stream changes and we're recording
    if (activeStream && context && isRecording) {
      startVisualization(context, activeStream);
      setIsInitialized(true);
    } else if (!isRecording && isInitialized) {
      stopVisualization();
    }
    
    // Clean up on unmount
    return () => {
      stopVisualization();
      cleanupAudioNodes();
    };
  }, [stream, audioContext, isRecording, isInitialized]);
  
  const cleanupAudioNodes = () => {
    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch (e) {
        // Already disconnected, ignore
      }
      sourceRef.current = null;
    }
    
    analyserRef.current = null;
  };
  
  const startVisualization = (context: AudioContext, activeStream: MediaStream) => {
    if (!canvasRef.current) return;
    
    try {
      // Clean up previous nodes
      cleanupAudioNodes();
      
      // Create analyzer node
      const analyser = context.createAnalyser();
      analyser.fftSize = Math.max(barCount * 2, 32);
      analyser.smoothingTimeConstant = smoothingTimeConstant;
      analyserRef.current = analyser;
      
      // Create source from stream and connect to analyzer
      const source = context.createMediaStreamSource(activeStream);
      source.connect(analyser);
      sourceRef.current = source;
      
      // Start drawing visualization
      drawVisualization();
    } catch (error) {
      console.error('Error starting visualization:', error);
    }
  };
  
  const drawVisualization = () => {
    if (!canvasRef.current || !analyserRef.current) return;
    
    const canvas = canvasRef.current;
    const canvasCtx = canvas.getContext('2d');
    if (!canvasCtx) return;
    
    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      if (!isRecording) return;
      
      animationFrameRef.current = requestAnimationFrame(draw);
      
      // Get frequency data from analyzer
      analyser.getByteFrequencyData(dataArray);
      
      // Clear canvas
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Calculate and draw bars
      const barWidth = (canvas.width / Math.min(bufferLength, barCount));
      let x = 0;
      
      // Create gradient if barColor is an array
      let gradient: CanvasGradient | string = typeof barColor === 'string' 
        ? barColor 
        : canvasCtx.createLinearGradient(0, 0, 0, canvas.height);
      
      if (typeof barColor !== 'string' && barColor.length >= 2) {
        const gradientStops = barColor.length;
        for (let i = 0; i < gradientStops; i++) {
          (gradient as CanvasGradient).addColorStop(i / (gradientStops - 1), barColor[i]);
        }
      }
      
      canvasCtx.fillStyle = gradient;
      
      // Calculate step size for fewer bars
      const step = Math.ceil(bufferLength / barCount);
      
      // Draw each frequency bar
      for (let i = 0; i < bufferLength; i += step) {
        // Average a few frequency bands for smoother visualization
        let sum = 0;
        let count = 0;
        
        for (let j = 0; j < step && i + j < bufferLength; j++) {
          sum += dataArray[i + j];
          count++;
        }
        
        const average = sum / count;
        const barHeight = (average / 255) * canvas.height;
        
        // Apply some easing for smoother animation
        // Round edges of bars for a more polished look
        canvasCtx.beginPath();
        canvasCtx.roundRect(
          x, 
          canvas.height - barHeight, 
          barWidth - 1, 
          barHeight,
          [2]
        );
        canvasCtx.fill();
        
        x += barWidth;
      }
    };
    
    draw();
  };
  
  const stopVisualization = () => {
    // Cancel animation frame if active
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    // Clear canvas
    if (canvasRef.current) {
      const canvasCtx = canvasRef.current.getContext('2d');
      if (canvasCtx) {
        canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
  };
  
  return (
    <canvas 
      ref={canvasRef}
      width={width}
      height={height}
      className="audio-visualizer"
      style={{
        width: width,
        height: height,
        borderRadius: '4px',
        background: backgroundColor
      }}
    />
  );
};

export default AudioVisualizer; 