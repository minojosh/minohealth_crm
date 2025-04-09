import React from 'react';

interface PulsingMicrophoneProps {
  isRecording: boolean;
  isProcessing: boolean;
  onClick: () => void;
  size?: number;
  primaryColor?: string;
  pulseColor?: string;
}

const PulsingMicrophone: React.FC<PulsingMicrophoneProps> = ({
  isRecording,
  isProcessing,
  onClick,
  size = 80,
  primaryColor = '#1976d2',
  pulseColor = '#9c27b0'
}) => {
  return (
    <div className="relative inline-flex items-center justify-center">
      {/* Main button */}
      <button
        style={{
          width: `${size}px`,
          height: `${size}px`,
          border: '2px solid',
          borderColor: isRecording ? pulseColor : primaryColor,
          borderRadius: '50%',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          backgroundColor: 'transparent',
          cursor: isProcessing ? 'not-allowed' : 'pointer',
          opacity: isProcessing ? 0.7 : 1,
          transition: 'all 0.2s ease',
          position: 'relative',
          zIndex: 10
        }}
        onClick={onClick}
        disabled={isProcessing}
        aria-label={isRecording ? "Stop recording" : "Start recording"}
      >
        {isRecording ? (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="6" y="6" width="12" height="12" fill="currentColor" />
          </svg>
        ) : (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" fill="currentColor" />
            <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" fill="currentColor" />
          </svg>
        )}
      </button>
      
      {/* Pulse animations */}
      {isRecording && (
        <>
          <div 
            className="pulse-ring pulse-ring-1"
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              border: `2px solid ${pulseColor}`,
              borderRadius: '50%',
              animation: 'pulse 2s cubic-bezier(0.455, 0.03, 0.515, 0.955) infinite',
            }}
          />
          <div 
            className="pulse-ring pulse-ring-2"
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              border: `2px solid ${pulseColor}`,
              borderRadius: '50%',
              animation: 'pulse 2s cubic-bezier(0.455, 0.03, 0.515, 0.955) infinite 0.5s',
            }}
          />
          <div 
            className="pulse-ring pulse-ring-3"
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              border: `2px solid ${pulseColor}`,
              borderRadius: '50%',
              animation: 'pulse 2s cubic-bezier(0.455, 0.03, 0.515, 0.955) infinite 1s',
            }}
          />
        </>
      )}
      
      {/* Processing spinner */}
      {isProcessing && (
        <div
          style={{
            position: 'absolute',
            top: `-10px`,
            left: `-10px`,
            width: `${size + 20}px`,
            height: `${size + 20}px`,
            border: `2px solid ${primaryColor}`,
            borderRadius: '50%',
            borderTopColor: 'transparent',
            animation: 'spin 1s linear infinite',
            zIndex: -1,
          }}
        />
      )}
      
      <style jsx>{`
        @keyframes pulse {
          0% {
            transform: scale(1);
            opacity: 0.8;
          }
          70% {
            transform: scale(1.5);
            opacity: 0;
          }
          100% {
            transform: scale(1.5);
            opacity: 0;
          }
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default PulsingMicrophone; 