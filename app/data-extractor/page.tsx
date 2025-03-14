"use client";
import { useState, useEffect } from "react";
import { Card, CardBody } from "@heroui/card";
import { Button } from "@heroui/button";
import { Divider } from "@heroui/divider";
import { Textarea } from "@heroui/input";
import { Tabs, Tab } from "@heroui/tabs";
import { Table, TableHeader, TableColumn, TableBody, TableRow, TableCell } from "@heroui/table";
import { Modal, ModalContent, ModalHeader, ModalBody, ModalFooter } from "@heroui/modal";
import { useDisclosure } from "@heroui/react";
import CombinedVoiceRecorder from "../../components/voice/VoiceRecorder";
import { TranscriptionStatus } from "../api/types";
import { 
  extractData, 
  extractFromAudio, 
  getPatientDetails, 
  getAllPatients, 
  ExtractedDataResponse, 
  PatientDetails, 
  PatientDetailsResponse 
} from "./api";
import { audioService } from "../api/audio";

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
  const [activeTab, setActiveTab] = useState("extract");
  const [patients, setPatients] = useState<PatientDetails[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<PatientDetailsResponse | null>(null);
  const [isLoadingPatients, setIsLoadingPatients] = useState(false);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [audioData, setAudioData] = useState<string | null>(null);
  const [isExtractingFromAudio, setIsExtractingFromAudio] = useState(false);

  // Load patients on component mount
  useEffect(() => {
    if (activeTab === "patients") {
      loadPatients();
    }
  }, [activeTab]);

  const loadPatients = async () => {
    setIsLoadingPatients(true);
    try {
      const response = await getAllPatients();
      setPatients(response.patients);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load patients");
    } finally {
      setIsLoadingPatients(false);
    }
  };

  const handleTranscriptionUpdate = (text: string) => {
    setTranscription(text);
  };

  const handleStatusChange = (status: TranscriptionStatus) => {
    setRecordingStatus(status);
    
    // If recording has stopped and we have audio data, store it
    if (status.status === "done" && audioService.getLastAudioData()) {
      setAudioData(audioService.getLastAudioData());
    }
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
      
      // If a patient was created, refresh the patients list
      if (data.patient_id) {
        loadPatients();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to extract data");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleExtractFromAudio = async () => {
    if (!audioData) {
      setError("No audio data available to extract from");
      return;
    }

    setIsExtractingFromAudio(true);
    setError(null);

    try {
      const data = await extractFromAudio(audioData);
      setExtractedData(data);
      
      // If a patient was created, refresh the patients list
      if (data.patient_id) {
        loadPatients();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to extract data from audio");
    } finally {
      setIsExtractingFromAudio(false);
    }
  };

  const handleViewPatient = async (patientId: number) => {
    try {
      const patientDetails = await getPatientDetails(patientId);
      setSelectedPatient(patientDetails);
      onOpen();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load patient details");
    }
  };

  const renderExtractedData = (data: ExtractedDataResponse) => {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="bg-gray-800 border-none shadow-lg overflow-hidden">
          <CardBody className="p-0">
            <div className="bg-blue-600 px-4 py-3">
              <h3 className="font-semibold text-white">Patient Information</h3>
            </div>
            <div className="p-4 space-y-2">
              {data.name && (
                <div className="flex items-center">
                  <span className="w-1/3 text-gray-400 text-sm">Name</span>
                  <span className="w-2/3 text-white font-medium">{data.name}</span>
                </div>
              )}
              {data.dob && (
                <div className="flex items-center">
                  <span className="w-1/3 text-gray-400 text-sm">Date of Birth</span>
                  <span className="w-2/3 text-white font-medium">{data.dob}</span>
                </div>
              )}
              {data.address && (
                <div className="flex items-center">
                  <span className="w-1/3 text-gray-400 text-sm">Address</span>
                  <span className="w-2/3 text-white font-medium">{data.address}</span>
                </div>
              )}
              {data.phone && (
                <div className="flex items-center">
                  <span className="w-1/3 text-gray-400 text-sm">Phone</span>
                  <span className="w-2/3 text-white font-medium">{data.phone}</span>
                </div>
              )}
              {data.email && (
                <div className="flex items-center">
                  <span className="w-1/3 text-gray-400 text-sm">Email</span>
                  <span className="w-2/3 text-white font-medium">{data.email}</span>
                </div>
              )}
              {data.insurance && (
                <div className="flex items-center">
                  <span className="w-1/3 text-gray-400 text-sm">Insurance</span>
                  <span className="w-2/3 text-white font-medium">{data.insurance}</span>
                </div>
              )}
              {data.patient_id && (
                <div className="flex items-center">
                  <span className="w-1/3 text-gray-400 text-sm">Patient ID</span>
                  <span className="w-2/3 text-white font-medium">{data.patient_id}</span>
                </div>
              )}
              {!data.name && !data.dob && !data.address && !data.phone && !data.insurance && (
                <div className="text-gray-400 text-center py-4">No patient information extracted</div>
              )}
            </div>
          </CardBody>
        </Card>

        <Card className="bg-gray-800 border-none shadow-lg overflow-hidden">
          <CardBody className="p-0">
            <div className="bg-purple-600 px-4 py-3">
              <h3 className="font-semibold text-white">Medical Information</h3>
            </div>
            <div className="p-4 space-y-2">
              {data.condition && (
                <div className="flex items-center">
                  <span className="w-1/3 text-gray-400 text-sm">Condition</span>
                  <span className="w-2/3 text-white font-medium">{data.condition}</span>
                </div>
              )}
              {data.symptoms && data.symptoms.length > 0 && (
                <div className="flex">
                  <span className="w-1/3 text-gray-400 text-sm">Symptoms</span>
                  <div className="w-2/3">
                    <div className="flex flex-wrap gap-2">
                      {data.symptoms.map((symptom, index) => (
                        <span key={index} className="bg-purple-900/50 text-purple-200 text-xs px-2 py-1 rounded-full">
                          {symptom}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
              {data.reason_for_visit && (
                <div className="flex items-start">
                  <span className="w-1/3 text-gray-400 text-sm">Reason for Visit</span>
                  <span className="w-2/3 text-white font-medium">{data.reason_for_visit}</span>
                </div>
              )}
              {!data.condition && (!data.symptoms || data.symptoms.length === 0) && !data.reason_for_visit && (
                <div className="text-gray-400 text-center py-4">No medical information extracted</div>
              )}
            </div>
          </CardBody>
        </Card>

        <Card className="bg-gray-800 border-none shadow-lg overflow-hidden">
          <CardBody className="p-0">
            <div className="bg-green-600 px-4 py-3">
              <h3 className="font-semibold text-white">Appointment Details</h3>
            </div>
            <div className="p-4 space-y-2">
              {data.appointment_details ? (
                <>
                  {data.appointment_details.type && (
                    <div className="flex items-center">
                      <span className="w-1/3 text-gray-400 text-sm">Type</span>
                      <span className="w-2/3 text-white font-medium">{data.appointment_details.type}</span>
                    </div>
                  )}
                  {data.appointment_details.doctor && (
                    <div className="flex items-center">
                      <span className="w-1/3 text-gray-400 text-sm">Doctor</span>
                      <span className="w-2/3 text-white font-medium">{data.appointment_details.doctor}</span>
                    </div>
                  )}
                  {data.appointment_details.scheduled_date && (
                    <div className="flex items-center">
                      <span className="w-1/3 text-gray-400 text-sm">Date</span>
                      <span className="w-2/3 text-white font-medium">{data.appointment_details.scheduled_date}</span>
                    </div>
                  )}
                  {data.appointment_details.time && (
                    <div className="flex items-center">
                      <span className="w-1/3 text-gray-400 text-sm">Time</span>
                      <span className="w-2/3 text-white font-medium">{data.appointment_details.time}</span>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-gray-400 text-center py-4">No appointment details extracted</div>
              )}
              {data.metadata && (
                <div className="mt-6 pt-4 border-t border-gray-700 text-xs text-gray-500">
                  <div className="flex justify-between">
                    <span>Processed: {data.metadata.processed_at}</span>
                    <span>Version: {data.metadata.version}</span>
                  </div>
                </div>
              )}
            </div>
          </CardBody>
        </Card>
      </div>
    );
  };

  const renderPatientsList = () => {
    if (isLoadingPatients) {
      return (
        <div className="flex justify-center items-center py-8">
          <div className="w-10 h-10 border-4 border-t-primary border-r-transparent border-b-primary border-l-transparent rounded-full animate-spin"></div>
          <span className="ml-3 text-gray-400">Loading patients...</span>
        </div>
      );
    }

    if (patients.length === 0) {
      return (
        <div className="text-center py-8 text-gray-400">
          <svg className="w-16 h-16 mx-auto mb-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          <p className="text-lg">No patients found</p>
          <p className="text-sm mt-2">Extract data from voice or text to create patient records</p>
        </div>
      );
    }

    return (
      <Table aria-label="Patients list" className="mt-4">
        <TableHeader>
          <TableColumn>ID</TableColumn>
          <TableColumn>Name</TableColumn>
          <TableColumn>Phone</TableColumn>
          <TableColumn>Email</TableColumn>
          <TableColumn>Date of Birth</TableColumn>
          <TableColumn>Actions</TableColumn>
        </TableHeader>
        <TableBody>
          {patients.map((patient) => (
            <TableRow key={patient.patient_id}>
              <TableCell>
                <span className="font-medium text-white bg-gray-700 px-2 py-1 rounded-md">{patient.patient_id}</span>
              </TableCell>
              <TableCell>{patient.name}</TableCell>
              <TableCell>{patient.phone}</TableCell>
              <TableCell>{patient.email || '-'}</TableCell>
              <TableCell>{patient.date_of_birth || '-'}</TableCell>
              <TableCell>
                <Button 
                  size="sm" 
                  color="primary" 
                  onClick={() => handleViewPatient(patient.patient_id)}
                  className="rounded-full"
                >
                  View Details
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    );
  };

  const renderPatientDetails = () => {
    if (!selectedPatient) return null;

    return (
      <div className="space-y-6">
        <div className="bg-gray-800 rounded-xl p-5 shadow-lg">
          <h3 className="text-lg font-semibold mb-4 text-white flex items-center">
            <svg className="w-5 h-5 mr-2 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
            </svg>
            Patient Information
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-700/50 p-3 rounded-lg">
              <span className="text-gray-400 text-sm">Name</span>
              <p className="text-white font-medium">{selectedPatient.patient.name}</p>
            </div>
            <div className="bg-gray-700/50 p-3 rounded-lg">
              <span className="text-gray-400 text-sm">Phone</span>
              <p className="text-white font-medium">{selectedPatient.patient.phone}</p>
            </div>
            <div className="bg-gray-700/50 p-3 rounded-lg">
              <span className="text-gray-400 text-sm">Email</span>
              <p className="text-white font-medium">{selectedPatient.patient.email || '-'}</p>
            </div>
            <div className="bg-gray-700/50 p-3 rounded-lg">
              <span className="text-gray-400 text-sm">Address</span>
              <p className="text-white font-medium">{selectedPatient.patient.address || '-'}</p>
            </div>
            <div className="bg-gray-700/50 p-3 rounded-lg">
              <span className="text-gray-400 text-sm">Date of Birth</span>
              <p className="text-white font-medium">{selectedPatient.patient.date_of_birth || '-'}</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl p-5 shadow-lg">
          <h3 className="text-lg font-semibold mb-4 text-white flex items-center">
            <svg className="w-5 h-5 mr-2 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
            </svg>
            Appointments
          </h3>
          {selectedPatient.appointments.length > 0 ? (
            <div className="overflow-hidden rounded-lg border border-gray-700">
              <Table aria-label="Appointments" className="min-w-full">
                <TableHeader>
                  <TableColumn>Date/Time</TableColumn>
                  <TableColumn>Type</TableColumn>
                  <TableColumn>Doctor</TableColumn>
                  <TableColumn>Status</TableColumn>
                </TableHeader>
                <TableBody>
                  {selectedPatient.appointments.map((appointment) => (
                    <TableRow key={appointment.appointment_id}>
                      <TableCell>{appointment.datetime || '-'}</TableCell>
                      <TableCell>{appointment.appointment_type || '-'}</TableCell>
                      <TableCell>{appointment.doctor_name || '-'}</TableCell>
                      <TableCell>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          appointment.status === 'scheduled' ? 'bg-blue-900/50 text-blue-200' :
                          appointment.status === 'completed' ? 'bg-green-900/50 text-green-200' :
                          appointment.status === 'cancelled' ? 'bg-red-900/50 text-red-200' :
                          'bg-gray-900/50 text-gray-200'
                        }`}>
                          {appointment.status}
                        </span>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : (
            <div className="text-center py-6 bg-gray-700/30 rounded-lg">
              <svg className="w-10 h-10 mx-auto mb-2 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
              </svg>
              <p className="text-gray-400">No appointments found</p>
            </div>
          )}
        </div>

        <div className="bg-gray-800 rounded-xl p-5 shadow-lg">
          <h3 className="text-lg font-semibold mb-4 text-white flex items-center">
            <svg className="w-5 h-5 mr-2 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
            </svg>
            Medical Conditions
          </h3>
          {selectedPatient.medical_conditions.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {selectedPatient.medical_conditions.map((condition) => (
                <div key={condition.condition_id} className="bg-gray-700/40 p-4 rounded-lg">
                  <div className="font-medium text-white mb-2">{condition.name}</div>
                  <div className="text-sm text-gray-400">Recorded: {condition.date_recorded || 'Unknown'}</div>
                  {condition.notes && <div className="text-sm text-gray-300 mt-2">{condition.notes}</div>}
                  
                  {condition.symptoms.length > 0 && (
                    <div className="mt-3">
                      <div className="font-medium text-sm text-gray-300 mb-2">Symptoms:</div>
                      <div className="flex flex-wrap gap-2">
                        {condition.symptoms.map((symptom) => (
                          <span key={symptom.symptom_id} className="bg-purple-900/40 text-purple-200 text-xs px-2 py-1 rounded-full">
                            {symptom.name} {symptom.severity ? `(${symptom.severity})` : ''}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-6 bg-gray-700/30 rounded-lg">
              <svg className="w-10 h-10 mx-auto mb-2 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
              </svg>
              <p className="text-gray-400">No medical conditions found</p>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col gap-6 pb-8">
      <div className="bg-gradient-to-r from-blue-900 to-indigo-900 rounded-xl p-6 shadow-lg">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1 className="text-3xl font-bold text-white">Medical Data Management</h1>
            <p className="text-blue-200 mt-1">Extract and manage patient information from voice or text</p>
          </div>
          <div className="flex gap-3">
            {activeTab === "extract" && (
              <>
                <Button 
                  color="primary" 
                  onClick={handleExtractData} 
                  isDisabled={!transcription || recordingStatus.isRecording || isProcessing}
                  isLoading={isProcessing}
                  className="bg-white/10 backdrop-blur-sm border border-white/20 hover:bg-white/20"
                  startContent={
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                    </svg>
                  }
                >
                  {isProcessing ? "Extracting Data..." : "Extract from Text"}
                </Button>
                <Button 
                  color="secondary" 
                  onClick={handleExtractFromAudio} 
                  isDisabled={!audioData || isExtractingFromAudio}
                  isLoading={isExtractingFromAudio}
                  className="bg-purple-500/20 backdrop-blur-sm border border-purple-500/30 hover:bg-purple-500/30"
                  startContent={
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
                    </svg>
                  }
                >
                  {isExtractingFromAudio ? "Extracting..." : "Extract from Audio"}
                </Button>
              </>
            )}
            {activeTab === "patients" && (
              <Button 
                color="primary" 
                onClick={loadPatients}
                isLoading={isLoadingPatients}
                className="bg-white/10 backdrop-blur-sm border border-white/20 hover:bg-white/20"
                startContent={
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                  </svg>
                }
              >
                Refresh Patients
              </Button>
            )}
          </div>
        </div>
      </div>
      
      <Tabs 
        selectedKey={activeTab} 
        onSelectionChange={(key) => setActiveTab(key as string)}
        className="bg-gray-900 p-2 rounded-xl"
        variant="underlined"
        color="primary"
      >
        <Tab 
          key="extract" 
          title={
            <div className="flex items-center gap-2 px-1">
              <svg className="w-5 h-5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
              </svg>
              Data Extraction
            </div>
          } 
        >
          <div className="mt-6">
            <Card className="bg-gray-800 border-none shadow-lg overflow-hidden">
              <CardBody>
                <div className="flex flex-col gap-4">
                  <h2 className="text-xl font-semibold text-white flex items-center">
                    <svg className="w-5 h-5 mr-2 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
                    </svg>
                    Voice Input & Transcription
                  </h2>
                  <div className="grid grid-cols-1 gap-4">
                    <CombinedVoiceRecorder 
                      onTranscriptionUpdate={handleTranscriptionUpdate}
                      onRecordingStatusChange={handleStatusChange}
                    />
                    <div className="bg-gray-900 rounded-xl p-4 min-h-[160px] shadow-inner relative">
                      {transcription ? (
                        <p className="text-gray-300 whitespace-pre-wrap">{transcription}</p>
                      ) : (
                        <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                          <p>Transcribed text will appear here...</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </CardBody>
            </Card>
          </div>

          <Card className="mt-6 bg-gray-800 border-none shadow-lg overflow-hidden">
            <CardBody>
              <div className="flex flex-col gap-4">
                <h2 className="text-xl font-semibold text-white flex items-center">
                  <svg className="w-5 h-5 mr-2 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                  </svg>
                  Extracted Data
                </h2>
                {error && (
                  <div className="bg-red-900/30 border border-red-500 text-red-300 text-sm p-4 rounded-lg flex items-start">
                    <svg className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <span>{error}</span>
                  </div>
                )}
                {extractedData ? (
                  renderExtractedData(extractedData)
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <Card className="bg-gray-700/30 border-none shadow-lg overflow-hidden">
                      <CardBody className="p-0">
                        <div className="bg-blue-600/50 px-4 py-3">
                          <h3 className="font-semibold text-white">Patient Information</h3>
                        </div>
                        <div className="p-6 flex items-center justify-center">
                          <div className="text-gray-400 text-center">
                            <svg className="w-10 h-10 mx-auto mb-2 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
                            </svg>
                            <p>No data extracted yet</p>
                          </div>
                        </div>
                      </CardBody>
                    </Card>
                    <Card className="bg-gray-700/30 border-none shadow-lg overflow-hidden">
                      <CardBody className="p-0">
                        <div className="bg-purple-600/50 px-4 py-3">
                          <h3 className="font-semibold text-white">Medical Information</h3>
                        </div>
                        <div className="p-6 flex items-center justify-center">
                          <div className="text-gray-400 text-center">
                            <svg className="w-10 h-10 mx-auto mb-2 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                            </svg>
                            <p>No data extracted yet</p>
                          </div>
                        </div>
                      </CardBody>
                    </Card>
                    <Card className="bg-gray-700/30 border-none shadow-lg overflow-hidden">
                      <CardBody className="p-0">
                        <div className="bg-green-600/50 px-4 py-3">
                          <h3 className="font-semibold text-white">Appointment Details</h3>
                        </div>
                        <div className="p-6 flex items-center justify-center">
                          <div className="text-gray-400 text-center">
                            <svg className="w-10 h-10 mx-auto mb-2 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                            </svg>
                            <p>No data extracted yet</p>
                          </div>
                        </div>
                      </CardBody>
                    </Card>
                  </div>
                )}
              </div>
            </CardBody>
          </Card>
        </Tab>
        
        <Tab 
          key="patients" 
          title={
            <div className="flex items-center gap-2 px-1">
              <svg className="w-5 h-5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path>
              </svg>
              Patient Records
            </div>
          }
        >
          <div className="mt-6">
            <Card className="bg-gray-800 border-none shadow-lg overflow-hidden">
              <CardBody>
                <div className="flex flex-col gap-4">
                  <h2 className="text-xl font-semibold text-white flex items-center">
                    <svg className="w-5 h-5 mr-2 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path>
                    </svg>
                    Patient Records
                  </h2>
                  {error && (
                    <div className="bg-red-900/30 border border-red-500 text-red-300 text-sm p-4 rounded-lg flex items-start">
                      <svg className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                      </svg>
                      <span>{error}</span>
                    </div>
                  )}
                  {renderPatientsList()}
                </div>
              </CardBody>
            </Card>
          </div>
        </Tab>
      </Tabs>
      
      <Modal 
        isOpen={isOpen} 
        onClose={onClose} 
        size="3xl"
        classNames={{
          base: "bg-gray-900 text-white border border-gray-700 shadow-xl",
          header: "border-b border-gray-700",
          body: "p-6",
          footer: "border-t border-gray-700"
        }}
      >
        <ModalContent>
          <ModalHeader>
            <h2 className="text-xl font-bold flex items-center">
              <svg className="w-6 h-6 mr-2 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
              </svg>
              {selectedPatient ? selectedPatient.patient.name : 'Patient Details'}
            </h2>
          </ModalHeader>
          <ModalBody>
            {selectedPatient && renderPatientDetails()}
          </ModalBody>
          <ModalFooter>
            <Button color="danger" variant="light" onClick={onClose}>
              Close
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  );
}