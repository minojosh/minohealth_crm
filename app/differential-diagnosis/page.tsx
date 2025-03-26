"use client";

import DiagnosisConversation from "@/components/differential-diagnosis/DiagnosisConversation";
import DiagnosisSummary from "@/components/differential-diagnosis/DiagnosisSummary";
import React, { useState } from "react";
import { useRouter } from 'next/navigation';

export default function Page() {
  const router = useRouter();
  return (
    <div className="h-[calc(100vh-160px)]">
        <>
              <DiagnosisConversation 
            patientId={1} // Using a default patientId of 1
            onBack={() => router.push('/')} // Redirecting to the differential-diagnosis page
          />
        </>
{/*   
        <DiagnosisSummary
          diagnosisResult={{
            primaryDiagnosis: "Primary Diagnosis",
            differentialDiagnoses: [
              "Differential Diagnosis 1",
              "Differential Diagnosis 2",
            ],
            recommendedTests: ["Test 1", "Test 2"],
            fullSummary: "Full Summary",
            patientId: 1,
          }}
        />
      )} */}
    </div>
  );
}
