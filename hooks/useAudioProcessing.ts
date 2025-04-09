import { useCallback } from 'react';

interface AudioProcessingOptions {
  noiseGateThreshold?: number;
  compressionThreshold?: number;
  compressionRatio?: number;
  normalizeLevel?: number;
  minAudioLevel?: number;
}

interface ProcessedAudio {
  base64Data: string;
  byteLength: number;
  duration: number;
  averageLevel: number;
}

export const useAudioProcessing = (options: AudioProcessingOptions = {}) => {
  const {
    noiseGateThreshold = 0.005,
    compressionThreshold = 0.3,
    compressionRatio = 0.6,
    normalizeLevel = 0.98,
    minAudioLevel = 0.001
  } = options;
  
  const processAudioBuffer = useCallback((audioBuffer: Float32Array[]): ProcessedAudio | null => {
    if (!audioBuffer.length) {
      console.warn("No audio buffer to process");
      return null;
    }
    
    try {
      // Concatenate all audio chunks
      const totalLength = audioBuffer.reduce((acc, chunk) => acc + chunk.length, 0);
      console.log(`Processing audio buffer: ${totalLength} samples (${totalLength/16000} seconds)`);
      
      if (totalLength === 0) {
        console.warn('No audio data to process');
        return null;
      }
      
      const concatenated = new Float32Array(totalLength);
      let offset = 0;
      
      for (const chunk of audioBuffer) {
        concatenated.set(chunk, offset);
        offset += chunk.length;
      }

      // Apply noise gate to remove background noise
      for (let i = 0; i < concatenated.length; i++) {
        if (Math.abs(concatenated[i]) < noiseGateThreshold) {
          concatenated[i] = 0;
        }
      }
      
      // Find maximum absolute value for normalization
      let maxAbs = 0;
      for (let i = 0; i < concatenated.length; i++) {
        maxAbs = Math.max(maxAbs, Math.abs(concatenated[i]));
      }
      
      if (maxAbs < 0.01) {
        console.warn("Audio level very low, applying minimum threshold");
        maxAbs = 0.01;
      }
      
      // Apply normalization and dynamic range compression
      for (let i = 0; i < concatenated.length; i++) {
        // Normalize to [-1, 1] range
        let normalized = concatenated[i] / maxAbs;
        
        // Apply dynamic range compression
        if (Math.abs(normalized) > compressionThreshold) {
          const overThreshold = Math.abs(normalized) - compressionThreshold;
          const compressed = overThreshold * compressionRatio;
          normalized = (normalized > 0) ? 
            compressionThreshold + compressed : 
            -compressionThreshold - compressed;
        }
        
        // Apply final normalization level
        concatenated[i] = normalized * normalizeLevel;
      }
      
      // Calculate average level for quality check
      let sumAbs = 0;
      for (let i = 0; i < concatenated.length; i++) {
        sumAbs += Math.abs(concatenated[i]);
      }
      const avgLevel = sumAbs / concatenated.length;
      
      // Server requires minimum level
      let finalAudio = concatenated;
      if (avgLevel < minAudioLevel) {
        console.warn(`Audio level too low (${avgLevel}), applying boost`);
        
        // Create new array with boosted values
        finalAudio = new Float32Array(concatenated.length);
        const boostFactor = minAudioLevel * 2 / avgLevel;
        
        for (let i = 0; i < concatenated.length; i++) {
          finalAudio[i] = concatenated[i] * boostFactor;
        }
      }
      
      // Convert the Float32Array to base64
      // First convert to 16-bit PCM
      const int16Data = new Int16Array(finalAudio.length);
      for (let i = 0; i < finalAudio.length; i++) {
        int16Data[i] = Math.max(-32768, Math.min(32767, finalAudio[i] * 32767));
      }
      
      // Convert to base64 efficiently in chunks to avoid stack overflow
      let base64Data = '';
      const chunkSize = 24 * 1024; // Process in chunks to avoid stack overflow
      
      for (let i = 0; i < int16Data.length; i += chunkSize) {
        const chunk = new Uint8Array(int16Data.buffer.slice(
          i * 2, 
          Math.min((i + chunkSize) * 2, int16Data.length * 2)
        ));
        
        for (let j = 0; j < chunk.length; j++) {
          base64Data += String.fromCharCode(chunk[j]);
        }
      }
      
      const final = btoa(base64Data);
      
      return {
        base64Data: final,
        byteLength: final.length,
        duration: totalLength / 16000, // assuming 16kHz sample rate
        averageLevel: avgLevel
      };
      
    } catch (error) {
      console.error("Error processing audio buffer:", error);
      return null;
    }
  }, [noiseGateThreshold, compressionThreshold, compressionRatio, normalizeLevel, minAudioLevel]);
  
  return {
    processAudioBuffer
  };
}; 