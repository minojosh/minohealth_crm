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
import VoiceRecorder from "../../components/voice/VoiceRecorder";
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
            {data.email && (
              <p className="text-sm"><span className="font-medium">Email</span>: {data.email}</p>
            )}
            {data.insurance && (
              <p className="text-sm"><span className="font-medium">Insurance</span>: {data.insurance}</p>
            )}
            {data.patient_id && (
              <p className="text-sm"><span className="font-medium">Patient ID</span>: {data.patient_id}</p>
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

  const renderPatientsList = () => {
    if (isLoadingPatients) {
      return <div className="text-center py-4">Loading patients...</div>;
    }

    if (patients.length === 0) {
      return <div className="text-center py-4">No patients found</div>;
    }

    return (
      <Table aria-label="Patients list">
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
              <TableCell>{patient.patient_id}</TableCell>
              <TableCell>{patient.name}</TableCell>
              <TableCell>{patient.phone}</TableCell>
              <TableCell>{patient.email || '-'}</TableCell>
              <TableCell>{patient.date_of_birth || '-'}</TableCell>
              <TableCell>
                <Button 
                  size="sm" 
                  color="primary" 
                  onClick={() => handleViewPatient(patient.patient_id)}
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
      <div className="space-y-4">
        <div>
          <h3 className="text-lg font-semibold mb-2">Patient Information</h3>
          <div className="grid grid-cols-2 gap-2">
            <div><span className="font-medium">Name:</span> {selectedPatient.patient.name}</div>
            <div><span className="font-medium">Phone:</span> {selectedPatient.patient.phone}</div>
            <div><span className="font-medium">Email:</span> {selectedPatient.patient.email || '-'}</div>
            <div><span className="font-medium">Address:</span> {selectedPatient.patient.address || '-'}</div>
            <div><span className="font-medium">Date of Birth:</span> {selectedPatient.patient.date_of_birth || '-'}</div>
          </div>
        </div>

        <div>
          <h3 className="text-lg font-semibold mb-2">Appointments</h3>
          {selectedPatient.appointments.length > 0 ? (
            <Table aria-label="Appointments">
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
                    <TableCell>{appointment.status}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="text-sm text-default-500">No appointments found</div>
          )}
        </div>

        <div>
          <h3 className="text-lg font-semibold mb-2">Medical Conditions</h3>
          {selectedPatient.medical_conditions.length > 0 ? (
            <div className="space-y-4">
              {selectedPatient.medical_conditions.map((condition) => (
                <div key={condition.condition_id} className="border p-3 rounded-lg">
                  <div className="font-medium">{condition.name}</div>
                  <div className="text-sm">Recorded: {condition.date_recorded || 'Unknown'}</div>
                  {condition.notes && <div className="text-sm">Notes: {condition.notes}</div>}
                  
                  {condition.symptoms.length > 0 && (
                    <div className="mt-2">
                      <div className="font-medium text-sm">Symptoms:</div>
                      <ul className="list-disc list-inside text-sm">
                        {condition.symptoms.map((symptom) => (
                          <li key={symptom.symptom_id}>
                            {symptom.name} {symptom.severity ? `(${symptom.severity})` : ''}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-sm text-default-500">No medical conditions found</div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Medical Data Management</h1>
        <div className="flex gap-2">
          {activeTab === "extract" && (
            <>
              <Button 
                color="primary" 
                onClick={handleExtractData} 
                isDisabled={!transcription || recordingStatus.isRecording || isProcessing}
                isLoading={isProcessing}
              >
                {isProcessing ? "Extracting Data..." : "Extract from Text"}
              </Button>
              <Button 
                color="secondary" 
                onClick={handleExtractFromAudio} 
                isDisabled={!audioData || isExtractingFromAudio}
                isLoading={isExtractingFromAudio}
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
            >
              Refresh Patients
            </Button>
          )}
        </div>
      </div>
      <Divider />
      
      <Tabs 
        selectedKey={activeTab} 
        onSelectionChange={(key) => setActiveTab(key as string)}
      >
        <Tab key="extract" title="Data Extraction">
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
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

          <Card className="mt-4">
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
        </Tab>
        
        <Tab key="patients" title="Patient Records">
          <div className="mt-4">
            <Card>
              <CardBody>
                <div className="flex flex-col gap-4">
                  <h2 className="text-xl font-semibold">Patient Records</h2>
                  {error && (
                    <div className="text-red-500 mb-4 p-4 bg-red-50 rounded-lg">
                      {error}
                    </div>
                  )}
                  {renderPatientsList()}
                </div>
              </CardBody>
            </Card>
          </div>
        </Tab>
      </Tabs>
      
      <Modal isOpen={isOpen} onClose={onClose} size="3xl">
        <ModalContent>
          <ModalHeader>
            <h2 className="text-xl font-bold">
              {selectedPatient ? `Patient: ${selectedPatient.patient.name}` : 'Patient Details'}
            </h2>
          </ModalHeader>
          <ModalBody>
            {renderPatientDetails()}
          </ModalBody>
          <ModalFooter>
            <Button color="primary" onClick={onClose}>
              Close
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  );
}