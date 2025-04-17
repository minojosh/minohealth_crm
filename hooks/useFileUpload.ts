import React, { useState, useCallback } from 'react';
import { ExtractedDataResponse, extractData } from '../app/data-extractor/api'; // Adjust path if needed
import { API_ENDPOINTS } from '../app/api/api';

export const useFileUpload = () => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [isProcessingUpload, setIsProcessingUpload] = useState(false);
    const [uploadError, setUploadError] = useState<string | null>(null);
    const [uploadTranscription, setUploadTranscription] = useState<string | null>(null);
    const [uploadExtractedData, setUploadExtractedData] = useState<ExtractedDataResponse | null>(null);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            // Basic validation (can be expanded)
            if (file.type === 'audio/wav' || file.type === 'audio/mpeg' || file.type === 'audio/webm' || file.type === 'audio/flac' || file.type === 'audio/mp4') { // Added more types based on backend
                setSelectedFile(file);
                setUploadError(null);
                // Reset results when new file is selected
                setUploadTranscription(null);
                setUploadExtractedData(null);
            } else {
                setSelectedFile(null);
                setUploadError('Please select a valid audio file (WAV, MP3, FLAC, MP4, WebM).');
            }
        } else {
            setSelectedFile(null);
        }
    };

    const processUploadedFile = useCallback(async () => {
        if (!selectedFile) {
            setUploadError('No file selected');
            return;
        }

        setIsProcessingUpload(true);
        setUploadError(null);
        setUploadTranscription(null);
        setUploadExtractedData(null);

        const formData = new FormData();
        formData.append('audio_file', selectedFile, selectedFile.name);

        try {
            // --- Send File to /upload-transcribe/ endpoint ---
            console.log(`Uploading file '${selectedFile.name}' to /upload-transcribe/...`);
            // Construct the full URL for the upload endpoint
            const uploadUrl = API_ENDPOINTS.uploadTranscribe; // Use the correct endpoint
            if (!uploadUrl) {
                throw new Error("Upload transcription endpoint URL is not defined in API_ENDPOINTS.");
            }

            const uploadResponse = await fetch(uploadUrl, {
                method: 'POST',
                body: formData,
                // Note: Don't set Content-Type header manually for FormData,
                // the browser will set it correctly with the boundary.
            });

            if (!uploadResponse.ok) {
                let errorDetail = `Upload API error: ${uploadResponse.status}`;
                try {
                    const errorJson = await uploadResponse.json();
                    errorDetail = errorJson.detail || errorDetail;
                } catch {
                    // Ignore if response is not JSON
                     errorDetail = `${errorDetail} - ${await uploadResponse.text()}`;
                }
                throw new Error(errorDetail);
            }

            const uploadResult = await uploadResponse.json();
            const transcription = uploadResult.transcription?.trim();

            if (!transcription || ['thank you.', 'thank you'].includes(transcription.toLowerCase())) {
                console.warn("Received potentially invalid transcription from upload:", transcription);
                // Decide if this is an error or just needs presenting differently
                // For now, treat as potentially valid but maybe log/flag it.
                // throw new Error('Transcription failed or returned invalid result.');
            }

            if (!transcription) {
                 throw new Error('Transcription result missing from upload response.');
            }

            setUploadTranscription(transcription);
            console.log('Transcription received from upload:', transcription.substring(0, 100) + '...');

            // --- Call extraction endpoint with the received transcription ---
            console.log('Sending transcription for data extraction...');
            const extracted = await extractData(transcription);
            setUploadExtractedData(extracted);
            console.log('Extraction complete.');
            // -------------------------------------------------------------

        } catch (error) {
            console.error('Error processing uploaded file:', error);
            const message = error instanceof Error ? error.message : 'Unknown error';
            setUploadError(`Failed to process file: ${message}`);
            // Ensure transcription/data are cleared on error
            setUploadTranscription(null);
            setUploadExtractedData(null);
        } finally {
            setIsProcessingUpload(false);
        }
    }, [selectedFile]);

    return {
        selectedFile,
        isProcessingUpload,
        uploadError,
        uploadTranscription,
        uploadExtractedData,
        handleFileChange,
        processUploadedFile,
        setUploadError // Expose setter to clear errors
    };
}; 