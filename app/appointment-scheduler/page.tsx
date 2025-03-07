"use client";

import { useState } from "react";
import { Card, CardBody } from "@heroui/card";
import { Button } from "@heroui/button";
import { Divider } from "@heroui/divider";
import { Textarea } from "@heroui/input";
import VoiceRecorder from "../../components/voice/VoiceRecorder";
import { TranscriptionStatus } from "../api/types";

interface AppointmentData {
  patientName?: string;
  appointmentDate?: string;
  appointmentTime?: string;
  reason?: string;
  doctorName?: string;
  notes?: string;
}

export default function AppointmentScheduler() {
  const [transcription, setTranscription] = useState("");
  const [recordingStatus, setRecordingStatus] = useState<TranscriptionStatus>({
    isRecording: false,
    duration: 0,
    status: "idle",
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [appointmentData, setAppointmentData] = useState<AppointmentData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleTranscriptionUpdate = (text: string) => {
    setTranscription(text);
  };

  const handleStatusChange = (status: TranscriptionStatus) => {
    setRecordingStatus(status);
  };

  const handleSubmit = async () => {
    if (!transcription.trim()) {
      setError("No transcription available to process");
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      // In a real implementation, this would call an API endpoint
      // that processes the transcription and extracts appointment data
      // For now, we'll simulate a response after a delay
      
      setTimeout(() => {
        // Simulated appointment data extraction
        const data: AppointmentData = {
          patientName: "John Doe",
          appointmentDate: "2023-12-15",
          appointmentTime: "10:30 AM",
          reason: "Follow-up consultation",
          doctorName: "Dr. Sarah Johnson",
          notes: "Patient needs to bring previous test results"
        };

        setAppointmentData(data);
        setIsProcessing(false);
      }, 1500);

    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to process appointment data");
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Appointment Scheduler</h1>
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
      
      <div className="flex justify-end mt-2">
        <Button 
          color="primary" 
          onClick={handleSubmit}
          isDisabled={!transcription || recordingStatus.isRecording || isProcessing}
          isLoading={isProcessing}
        >
          {isProcessing ? "Processing..." : "Schedule Appointment"}
        </Button>
      </div>
      
      {error && (
        <div className="text-red-500 mt-2">{error}</div>
      )}
      
      {appointmentData && (
        <Card className="mt-4">
          <CardBody>
            <div className="flex flex-col gap-4">
              <h2 className="text-xl font-semibold">Appointment Details</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium">Patient Name</p>
                  <p className="text-default-700">{appointmentData.patientName}</p>
                </div>
                <div>
                  <p className="text-sm font-medium">Doctor</p>
                  <p className="text-default-700">{appointmentData.doctorName}</p>
                </div>
                <div>
                  <p className="text-sm font-medium">Date</p>
                  <p className="text-default-700">{appointmentData.appointmentDate}</p>
                </div>
                <div>
                  <p className="text-sm font-medium">Time</p>
                  <p className="text-default-700">{appointmentData.appointmentTime}</p>
                </div>
                <div className="col-span-2">
                  <p className="text-sm font-medium">Reason</p>
                  <p className="text-default-700">{appointmentData.reason}</p>
                </div>
                {appointmentData.notes && (
                  <div className="col-span-2">
                    <p className="text-sm font-medium">Notes</p>
                    <p className="text-default-700">{appointmentData.notes}</p>
                  </div>
                )}
              </div>
            </div>
          </CardBody>
        </Card>
      )}
    </div>
  );
}