"use client";

import DiagnosisConversation from "@/components/differential-diagnosis/Conversation";  
import DiagnosisSummary from "@/components/differential-diagnosis/Summary";
// import DiagnosisConversation from "@/components/differential-diagnosis/DiagnosisConversation";
// import DiagnosisSummary from "@/components/differential-diagnosis/DiagnosisSummary";
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
    </div>
  );
}
