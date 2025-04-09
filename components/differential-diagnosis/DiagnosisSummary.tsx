import React from 'react';
import { Card, CardBody, CardHeader } from "@heroui/card";
import { Badge } from "@heroui/badge";

interface DiagnosisResult {
  primaryDiagnosis: string;
  differentialDiagnoses: string[];
  recommendedTests: string[];
  fullSummary: string;
  patientId: number;
  conditionId?: number;
}

interface DiagnosisSummaryProps {
  diagnosisResult: DiagnosisResult;
}

const DiagnosisSummary = ({ diagnosisResult }: DiagnosisSummaryProps) => {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-0">
          <h2 className="text-xl font-semibold">Diagnosis Summary</h2>
          <p className="text-default-500"></p>
        </CardHeader>
        <CardBody className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold mb-2">Primary Diagnosis</h3>
            <div className="p-3 bg-blue-100 border text-black border-blue-500 rounded-md">
              {diagnosisResult.primaryDiagnosis || "No primary diagnosis identified"}
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">Differential Diagnoses</h3>
            {diagnosisResult.differentialDiagnoses?.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {diagnosisResult.differentialDiagnoses.map((diagnosis, index) => (
                  <Badge key={index} variant="flat" className="px-3 py-1 rounded-md">
                    {diagnosis}
                  </Badge>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 italic">No differential diagnoses provided</p>
            )}
          </div>
          
          {diagnosisResult.recommendedTests?.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold mb-2">Recommended Tests</h3>
              <ul className="list-disc pl-5 space-y-1">
                {diagnosisResult.recommendedTests.map((test, index) => (
                  <li key={index}>{test}</li>
                ))}
              </ul>
            </div>
          )}
          
          <div>
            <h3 className="text-lg font-semibold mb-2">Full Assessment</h3>
            <div className="p-4 bg-gray-50 border border-gray-200 text-black rounded-md whitespace-pre-line">
              {diagnosisResult.fullSummary}
            </div>
          </div>
        </CardBody>
      </Card>
    </div>
  );
};

export default DiagnosisSummary; 