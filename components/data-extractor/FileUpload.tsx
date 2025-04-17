import React, { useState, useCallback, useRef } from 'react';
import { ExtractedDataResponse, extractData } from '../api';
import { Button } from "@heroui/button";
import { Input } from "@heroui/input"; // Use Input for file selection appearance
import { Progress } from "@heroui/progress";

interface FileUploadProps {
    onProcessComplete: (data: ExtractedDataResponse, transcription: string) => void;
    // Removed internal state management, will rely on hook state exposed via props if needed
    // Or parent component manages state
}

// Remove API logic - should be handled by parent/hook

const FileUpload: React.FC<FileUploadProps> = ({ onProcessComplete }) => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
             if (['audio/wav', 'audio/mpeg', 'audio/webm', 'audio/flac', 'audio/mp4'].includes(file.type)) {
                 setSelectedFile(file);
                 setError(null);
             } else {
                 setSelectedFile(null);
                 setError('Please select a valid audio file (WAV, MP3, FLAC, MP4, WebM).');
             }
        } else {
            setSelectedFile(null);
        }
        // Reset input value to allow re-selecting the same file
        if(event.target) event.target.value = '';
    };

    const triggerFileSelect = () => fileInputRef.current?.click();

    const processFile = useCallback(async () => {
        if (!selectedFile) {
            setError('No file selected');
            return;
        }

        setIsProcessing(true);
        setError(null);
        const formData = new FormData();
        formData.append('audio_file', selectedFile, selectedFile.name);

        try {
            // Assume API_ENDPOINTS is globally accessible or passed via props/context
            const { API_ENDPOINTS } = await import('../../api/api'); // Dynamic import

            console.log(`Uploading file '${selectedFile.name}' to ${API_ENDPOINTS.uploadTranscribe}...`);
            const uploadResponse = await fetch(API_ENDPOINTS.uploadTranscribe, {
                method: 'POST',
                body: formData,
            });

            if (!uploadResponse.ok) {
                 let errorDetail = `Upload API error: ${uploadResponse.status}`;
                 try {
                     const errorJson = await uploadResponse.json();
                     errorDetail = errorJson.detail || errorDetail;
                 } catch { errorDetail = `${errorDetail} - ${await uploadResponse.text()}` }
                 throw new Error(errorDetail);
            }

            const uploadResult = await uploadResponse.json();
            const transcription = uploadResult.transcription?.trim();

            if (!transcription) throw new Error('Transcription missing from upload response.');
            if (['thank you.', 'thank you'].includes(transcription.toLowerCase())) {
                 console.warn("Potentially invalid transcription:", transcription);
            }
            console.log('Transcription received:', transcription.substring(0, 100) + '...');

            console.log('Sending transcription for data extraction...');
            const extracted = await extractData(transcription);
            console.log('Extraction complete.');

            onProcessComplete(extracted, transcription);
            setSelectedFile(null); // Clear selection on success

        } catch (err) {
            console.error('Error processing uploaded file:', err);
            setError(`Processing failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
        } finally {
            setIsProcessing(false);
        }
    }, [selectedFile, onProcessComplete]);

    return (
        <div className="space-y-3">
            {/* Hidden file input */}
            <input
                ref={fileInputRef}
                type="file"
                accept="audio/wav,audio/mpeg,audio/webm,audio/flac,audio/mp4"
                onChange={handleFileChange}
                disabled={isProcessing}
                className="hidden"
            />

            {/* Button to trigger file selection */}
            <Button
                onPress={triggerFileSelect}
                isDisabled={isProcessing}
                variant="bordered"
                startContent={
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                         <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0 3 3m-3-3-3 3M6.75 19.5a4.5 4.5 0 0 1-1.41-8.775 5.25 5.25 0 0 1 10.233-2.33 3 3 0 0 1 3.758 3.848A3.752 3.752 0 0 1 18 19.5H6.75Z" />
                    </svg>
                }
            >
                {selectedFile ? `Selected: ${selectedFile.name}` : "Choose Audio File"}
            </Button>

            {/* Processing Button */}
            {selectedFile && (
                <Button
                    onPress={processFile}
                    isDisabled={!selectedFile || isProcessing}
                    color="secondary"
                    isLoading={isProcessing}
                >
                    {isProcessing ? 'Processing...' : 'Process Uploaded File'}
                </Button>
            )}

            {/* Progress and Error Display */}
            {isProcessing && (
                 <Progress
                     size="sm"
                     isIndeterminate
                     aria-label="Processing file..."
                     className="max-w-full mt-2"
                 />
            )}
            {error && <p className="text-red-500 text-sm mt-2">Error: {error}</p>}
        </div>
    );
};

export default FileUpload; 