"use client";

import { useState } from "react";
import { Card, CardBody } from "@heroui/card";
import { Button } from "@heroui/button";
import { Textarea } from "@heroui/input";
import { MicrophoneIcon } from "@heroicons/react/24/solid";
import { SchedulerConversation } from "@/components/scheduler/SchedulerConversation";

export default function AppointmentScheduler() {
  const [context, setContext] = useState("");
  const [patientId, setPatientId] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [showConversation, setShowConversation] = useState(false);

  const handleSendContext = async () => {
    if (!context.trim()) {
      setError("Please provide context for the appointment");
      return;
    }

    if (!patientId.trim()) {
      setError("Please provide a patient ID");
      return;
    }

    setShowConversation(true);
  };

  return (
    <div className="container mx-auto p-4">
      {!showConversation ? (
        <Card className="bg-gray-900 shadow-xl">
          <CardBody className="p-6">
            <h2 className="text-2xl font-semibold text-white mb-2">Appointment Scheduler</h2>
            
            {/* Patient ID Input */}
            <div className="mb-6">
              <label htmlFor="patientId" className="block text-gray-400 mb-2">
                Patient ID
              </label>
              <input
                id="patientId"
                type="text"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                placeholder="Enter patient ID"
                className="w-full bg-gray-800 border-0 rounded-xl text-gray-200 placeholder-gray-500 p-4 focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Context Input */}
            <div className="mb-6">
              <label htmlFor="context" className="block text-gray-400 mb-2">
                Appointment Details
              </label>
              <div className="relative">
                <Textarea
                  id="context"
                  value={context}
                  onChange={(e) => setContext(e.target.value)}
                  placeholder="Type your appointment requirements..."
                  className="w-full min-h-[120px] bg-gray-800 border-0 rounded-xl text-gray-200 placeholder-gray-500 p-4 focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>

            {error && (
              <div className="mb-4 p-3 bg-red-900/30 border border-red-800 rounded-lg text-red-400">
                <p>{error}</p>
              </div>
            )}

            <Button
              onClick={handleSendContext}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-xl"
            >
              Start Scheduling
            </Button>
          </CardBody>
        </Card>
      ) : (
        <div className="w-full">
          <SchedulerConversation
            initialContext={context}
            patientId={patientId}
            onComplete={(result) => {
              console.log("Scheduling completed:", result);
              setShowConversation(false);
            }}
          />
        </div>
      )}
    </div>
  );
}