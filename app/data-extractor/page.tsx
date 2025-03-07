"use client";
import { useState } from "react";
import { Card, CardBody } from "@heroui/card";
import { Button } from "@heroui/button";
import { Divider } from "@heroui/divider";
import { Textarea } from "@heroui/input";
import VoiceRecorder from "../../components/voice/VoiceRecorder";
import { TranscriptionStatus } from "../api/types";
import { extractData, ExtractedDataResponse } from "./api";

export default function DataExtractor() {
  const [transcription, setTranscription] = useState("");
  const [recordingStatus, setRecordingStatus] = useState<TranscriptionStatus>({
    isRecording: false,
    duration: 0,
    status: "idle",
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [extractedData, setExtractedData] = useState<ExtractedDataResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleTranscriptionUpdate = (text: string) => {
    setTranscription(text);
  };

  const handleStatusChange = (status: TranscriptionStatus) => {
    setRecordingStatus(status);
  };

  const handleExtractData = async () => {
    if (!transcription.trim()) {
      setError("No transcription available to extract data from");
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const data = await extractData(transcription);
      setExtractedData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to extract data");
    } finally {
      setIsProcessing(false);
    }
  };

  const renderExtractedData = (data: ExtractedDataResponse) => {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 border rounded-lg">
          <h3 className="font-medium mb-2">Patient Information</h3>
          <div className="space-y-1">
            {data.name && (
              <p className="text-sm"><span className="font-medium">Name</span>: {data.name}</p>
            )}
            {data.dob && (
              <p className="text-sm"><span className="font-medium">Date of Birth</span>: {data.dob}</p>
            )}
            {data.address && (
              <p className="text-sm"><span className="font-medium">Address</span>: {data.address}</p>
            )}
            {data.phone && (
              <p className="text-sm"><span className="font-medium">Phone</span>: {data.phone}</p>
            )}
            {data.insurance && (
              <p className="text-sm"><span className="font-medium">Insurance</span>: {data.insurance}</p>
            )}
            {!data.name && !data.dob && !data.address && !data.phone && !data.insurance && (
              <div className="text-sm text-default-500">No patient information extracted</div>
            )}
          </div>
        </div>

        <div className="p-4 border rounded-lg">
          <h3 className="font-medium mb-2">Medical Information</h3>
          <div className="space-y-1">
            {data.condition && (
              <p className="text-sm"><span className="font-medium">Condition</span>: {data.condition}</p>
            )}
            {data.symptoms && data.symptoms.length > 0 && (
              <p className="text-sm">
                <span className="font-medium">Symptoms</span>: {data.symptoms.join(", ")}
              </p>
            )}
            {data.reason_for_visit && (
              <p className="text-sm">
                <span className="font-medium">Reason for Visit</span>: {data.reason_for_visit}
              </p>
            )}
            {!data.condition && (!data.symptoms || data.symptoms.length === 0) && !data.reason_for_visit && (
              <div className="text-sm text-default-500">No medical information extracted</div>
            )}
          </div>
        </div>

        <div className="p-4 border rounded-lg">
          <h3 className="font-medium mb-2">Appointment Details</h3>
          {data.appointment_details ? (
            <div className="space-y-1">
              {data.appointment_details.type && (
                <p className="text-sm">
                  <span className="font-medium">Type</span>: {data.appointment_details.type}
                </p>
              )}
              {data.appointment_details.doctor && (
                <p className="text-sm">
                  <span className="font-medium">Doctor</span>: {data.appointment_details.doctor}
                </p>
              )}
              {data.appointment_details.scheduled_date && (
                <p className="text-sm">
                  <span className="font-medium">Date</span>: {data.appointment_details.scheduled_date}
                </p>
              )}
              {data.appointment_details.time && (
                <p className="text-sm">
                  <span className="font-medium">Time</span>: {data.appointment_details.time}
                </p>
              )}
            </div>
          ) : (
            <div className="text-sm text-default-500">No appointment details extracted</div>
          )}
          {data.metadata && (
            <div className="mt-4 pt-4 border-t text-xs text-default-400">
              <p>Processed: {data.metadata.processed_at}</p>
              <p>Version: {data.metadata.version}</p>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Data Extractor</h1>
        <Button 
          color="primary" 
          onClick={handleExtractData} 
          isDisabled={!transcription || recordingStatus.isRecording || isProcessing}
          isLoading={isProcessing}
        >
          {isProcessing ? "Extracting Data..." : "Extract Data"}
        </Button>
      </div>
      <Divider />
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardBody>
            <div className="flex flex-col gap-4">
              <h2 className="text-xl font-semibold">Voice Input</h2>
              <VoiceRecorder 
                onTranscriptionUpdate={handleTranscriptionUpdate}
                onRecordingStatusChange={handleStatusChange}
              />
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardBody>
            <div className="flex flex-col gap-4">
              <h2 className="text-xl font-semibold">Transcription</h2>
              <Textarea
                value={transcription}
                placeholder="Transcribed text will appear here..."
                className="min-h-[160px]"
                readOnly
              />
            </div>
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardBody>
          <div className="flex flex-col gap-4">
            <h2 className="text-xl font-semibold">Extracted Data</h2>
            {error && (
              <div className="text-red-500 mb-4 p-4 bg-red-50 rounded-lg">
                {error}
              </div>
            )}
            {extractedData ? (
              renderExtractedData(extractedData)
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 border rounded-lg">
                  <h3 className="font-medium mb-2">Patient Information</h3>
                  <div className="text-sm text-default-500">No data extracted yet</div>
                </div>
                <div className="p-4 border rounded-lg">
                  <h3 className="font-medium mb-2">Medical Information</h3>
                  <div className="text-sm text-default-500">No data extracted yet</div>
                </div>
                <div className="p-4 border rounded-lg">
                  <h3 className="font-medium mb-2">Appointment Details</h3>
                  <div className="text-sm text-default-500">No data extracted yet</div>
                </div>
              </div>
            )}
          </div>
        </CardBody>
      </Card>
    </div>
  );
}