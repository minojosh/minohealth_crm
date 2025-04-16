"use client";
import React, { useState, useCallback } from 'react';
import { Tabs, Tab } from "@heroui/tabs"; // Import HeroUI Tabs
import { Card, CardBody } from "@heroui/card"; // Import HeroUI Card
import { Button } from "@heroui/button"; // Import HeroUI Button
import { useAudioRecorder } from './hooks/useAudioRecorder';
import { useFileUpload } from './hooks/useFileUpload';
import RecorderControls from './components/RecorderControls';
import TranscriptionView from './components/TranscriptionView';
import FileUpload from './components/FileUpload';
import ExtractedDataDisplay from './components/ExtractedDataDisplay';
import PatientsTab from './components/PatientsTab';
import { ExtractedDataResponse, extractData } from './api';

const TABS = {
    EXTRACT: 'extract',
    PATIENTS: 'patients'
};

export default function DataExtractorPage() {
    const [activeTab, setActiveTab] = useState<React.Key>(TABS.EXTRACT);
    const [manualTranscription, setManualTranscription] = useState('');
    const [manualExtractedData, setManualExtractedData] = useState<ExtractedDataResponse | null>(null);
    const [isManualProcessing, setIsManualProcessing] = useState(false);
    const [manualError, setManualError] = useState<string | null>(null);

    // --- Audio Recorder Hook ---
    const {
        isRecording,
        isProcessing: isRecordingProcessing,
        error: recorderError,
        transcription: recorderTranscription,
        extractedData: recorderExtractedData,
        startRecording,
        stopRecording,
        setError: setRecorderError
    } = useAudioRecorder();

    const handleToggleRecording = () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    };

    // --- File Upload Hook ---
    const [fileExtractedData, setFileExtractedData] = useState<ExtractedDataResponse | null>(null);
    const [fileTranscription, setFileTranscription] = useState<string | null>(null);
    const handleFileUploadComplete = useCallback((data: ExtractedDataResponse, transcription: string) => {
        console.log("File processing complete in page:", data);
        setFileExtractedData(data);
        setFileTranscription(transcription);
        // Also update manual transcription field if user hasn't typed anything
        if (!manualTranscription) {
             setManualTranscription(transcription);
        }
    }, [manualTranscription]); // Depend on manualTranscription to avoid overwriting user input

    // --- Manual Text Extraction ---
    const handleExtractFromManualText = async () => {
        if (!manualTranscription.trim()) {
            setManualError("Please enter text to extract data from.");
            return;
        }
        setIsManualProcessing(true);
        setManualError(null);
        setManualExtractedData(null);
        try {
            const data = await extractData(manualTranscription);
            setManualExtractedData(data);
            // Clear other sources when manual extraction is successful
            setRecorderError(null);
            setFileExtractedData(null);
            setFileTranscription(null);
        } catch (err) {
            console.error("Error extracting from manual text:", err);
            setManualError(`Extraction failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
        } finally {
            setIsManualProcessing(false);
        }
    };

    // Consolidate data/transcription/error states
    let displayData: ExtractedDataResponse | null = null;
    let displayTranscription = manualTranscription; // Default to manual text
    let displayError = manualError || recorderError; // Combine errors

    if (recorderExtractedData) {
        displayData = recorderExtractedData;
        if (!manualTranscription) displayTranscription = recorderTranscription;
    } else if (fileExtractedData) {
        displayData = fileExtractedData;
        if (!manualTranscription) displayTranscription = fileTranscription || '';
    } else if (manualExtractedData) {
        displayData = manualExtractedData;
    }

    // Function to clear all results
    const clearAll = () => {
         setManualTranscription('');
         setManualExtractedData(null);
         setManualError(null);
         setRecorderError(null); // Assumes setError in hook clears internal state
         setFileExtractedData(null);
         setFileTranscription(null);
         // TODO: Potentially call a reset function within the hooks if they manage more state
    };

    return (
        // Use Tailwind CSS classes for overall layout and styling if available,
        // otherwise, rely on HeroUI component styling + minimal inline styles.
        <div className="container mx-auto px-4 py-8 max-w-4xl bg-gray-900 text-gray-100 font-sans">
            <h1 className="text-3xl font-bold text-center text-blue-400 mb-8">Data Extractor</h1>

            {/* Tab Navigation */}
            <div className="flex justify-center mb-6">
                <Tabs
                    selectedKey={activeTab}
                    onSelectionChange={setActiveTab}
                    aria-label="Data Extractor Tabs"
                    color="primary"
                    variant="underlined"
                    classNames={{
                        tabList: "bg-gray-800 rounded-lg p-1",
                        cursor: "bg-blue-600",
                        tab: "text-white px-4 py-2",
                        tabContent: "group-data-[selected=true]:text-white"
                    }}
                >
                    {/* <Tab key={TABS.EXTRACT} title="Extract Data" /> */}
                    {/* <Tab key={TABS.PATIENTS} title="View Patients" /> */}
                </Tabs>
            </div>

            {/* Global Error Display */}
             {displayError && (
                <Card className="mb-6 bg-red-900/50 border border-red-500">
                     <CardBody className="flex justify-between items-center text-red-100 p-3">
                         <span>Error: {displayError}</span>
                         <Button isIconOnly size="sm" variant="light" onPress={clearAll} className="text-red-100">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                            </svg>
                         </Button>
                     </CardBody>
                </Card>
            )}

            {/* Tab Content */}
            {activeTab === TABS.EXTRACT && (
                <Card className="bg-gray-800 border-none shadow-lg mb-6">
                     <CardBody className="p-6 space-y-6">
                        {/* Voice Input Section */}
                        <div>
                             <h2 className="text-xl font-semibold text-white mb-4">Voice Input</h2>
                             <RecorderControls
                                 isRecording={isRecording}
                                 isProcessing={isRecordingProcessing}
                                 onToggleRecording={handleToggleRecording}
                             />
                        </div>

                        {/* File Upload Section */}
                        <div>
                             <h2 className="text-xl font-semibold text-white mb-4">Or Upload Audio File</h2>
                             <FileUpload onProcessComplete={handleFileUploadComplete} />
                        </div>

                        {/* Divider */}
                        <hr className="border-gray-600" />

                        {/* Transcription & Manual Extraction Section */}
                        <div>
                             {/* <h2 className="text-xl font-semibold text-white mb-4">Transcription</h2> */}
                             <TranscriptionView
                                 transcription={displayTranscription}
                                 onTranscriptionChange={setManualTranscription}
                                 onClear={clearAll}
                                 isDisabled={isRecording || isRecordingProcessing}
                             />
                             <Button
                                 color="primary"
                                 onPress={handleExtractFromManualText}
                                 isDisabled={!manualTranscription.trim() || isManualProcessing || isRecordingProcessing}
                                 isLoading={isManualProcessing}
                                 className="mt-4"
                             >
                                 {isManualProcessing ? 'Extracting...' : 'Extract Data from Text'}
                             </Button>
                        </div>
                    </CardBody>
                </Card>
            )}

            {activeTab === TABS.PATIENTS && (
                <PatientsTab />
            )}

            {/* Extracted Data Display (always shown if data exists) */}
            {displayData && activeTab === TABS.EXTRACT && (
                 <ExtractedDataDisplay data={displayData} />
            )}

        </div>
    );
}