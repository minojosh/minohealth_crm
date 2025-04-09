"use client";
import { useState, useEffect, useRef } from "react";
import { Card, CardBody } from "@heroui/card";
import { Button } from "@heroui/button";
import { Divider } from "@heroui/divider";
import { Textarea } from "@heroui/input";
import { Tabs, Tab } from "@heroui/tabs";
import { Table, TableHeader, TableColumn, TableBody, TableRow, TableCell } from "@heroui/table";
import { Modal, ModalContent, ModalHeader, ModalBody, ModalFooter } from "@heroui/modal";
import { useDisclosure } from "@heroui/react";
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
  // Use useRef to track previous transcription state
  const prevTranscriptionRef = useRef("");
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
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessingUpload, setIsProcessingUpload] = useState(false);

  // Add a ref to track the MediaRecorder instance
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Add a ref to keep audio chunks
  const audioChunksRef = useRef<BlobPart[]>([]);

  // Add state for recording
  const [isRecording, setIsRecording] = useState(false);

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

  // Modified to correctly accumulate transcription chunks like in medical_assistant.py
  const handleTranscriptionUpdate = (text: string, sessionId?: string) => {
    if (!text) return; // Ignore empty updates
    
    // Track the session ID when provided
    if (sessionId && !currentSessionId) {
      setCurrentSessionId(sessionId);
      console.log(`Setting transcription session ID: ${sessionId}`);
    }
    
    setTranscription(prevText => {
      // If we already have a transcription and we're actively recording
      // then append the new text instead of replacing it
      if (prevText && recordingStatus.status === "recording") {
        // Store the previous text for comparison
        prevTranscriptionRef.current = prevText;
        
        // Check if the new chunk might be a duplicate or overlap
        if (!prevText.toLowerCase().includes(text.toLowerCase())) {
          return `${prevText} ${text}`;
        }
        return prevText;
      } else {
        // First chunk or not recording - just use the new text
        return text;
      }
    });
  };

  const handleStatusChange = (status: TranscriptionStatus) => {
    setRecordingStatus(status);
    
    // Reset transcription when starting a new recording
    if (status.status === "recording" && status.isRecording && !recordingStatus.isRecording) {
      setTranscription("");
      prevTranscriptionRef.current = "";
      
      // Clear previous session
      setCurrentSessionId(null);
    }
    
    // If recording has stopped and we have audio data, store it
    if (status.status === "done" && audioService.getLastAudioData()) {
      setAudioData(audioService.getLastAudioData());
      
      // Capture the session ID if available
      if (!currentSessionId) {
        const sessionId = audioService.getCurrentSessionId();
        if (sessionId) {
          setCurrentSessionId(sessionId);
          console.log(`Captured session ID after recording: ${sessionId}`);
        }
      }
    }
  };

  // Add clear transcription function
  const clearTranscription = () => {
    setTranscription("");
    prevTranscriptionRef.current = "";
  };

  const handleExtractData = async () => {
    if (!transcription.trim()) {
      setError("No transcription available to extract data from");
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      // Use the extractData function from the API module
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

  // Modified handleExtractFromAudio function using SchedulerConversation approach
  const handleExtractFromAudio = async () => {
    if (isRecording) {
      // If already recording, stop recording
      stopRecording();
      return;
    }
    
    setIsRecording(true);
    setError(null);
    setTranscription("");
    
    try {
      console.log("Requesting microphone access");
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,
          channelCount: 1
        }
      });
      
      console.log("Microphone access granted");
      streamRef.current = stream;
      
      // Create an AudioContext to process the audio
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      
      // Connect the audio nodes
      source.connect(processor);
      processor.connect(audioContext.destination);
      
      // Buffer to store raw audio data
      const audioBuffer: Float32Array[] = [];
      
      // Process audio data
      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        audioBuffer.push(new Float32Array(inputData));
      };
      
      // Create MediaRecorder for backup WebM recording
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 16000
      });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        console.log("Data available:", event.data.size);
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        console.log("MediaRecorder stopped, processing audio...");
        setIsProcessing(true);
        
        try {
          // Log initial audio buffer state
          console.log("Audio buffer chunks:", audioBuffer.length);
          
          // Concatenate all audio chunks
          const totalLength = audioBuffer.reduce((acc, chunk) => acc + chunk.length, 0);
          console.log(`Total audio length: ${totalLength} samples (${totalLength/16000} seconds)`);
          
          if (totalLength === 0) {
            console.warn('No audio data to process');
            setError('No audio data captured');
            setIsProcessing(false);
            setIsRecording(false);
            return;
          }
          
          const concatenated = new Float32Array(totalLength);
          let offset = 0;
          
          for (const chunk of audioBuffer) {
            concatenated.set(chunk, offset);
            offset += chunk.length;
          }

          // Find maximum absolute value for normalization
          let maxAbs = 0;
          for (let i = 0; i < concatenated.length; i++) {
            maxAbs = Math.max(maxAbs, Math.abs(concatenated[i]));
          }
          console.log("Maximum absolute value:", maxAbs);
          
          if (maxAbs > 0) {  // Avoid division by zero
            for (let i = 0; i < concatenated.length; i++) {
              concatenated[i] = concatenated[i] / maxAbs;
            }
          } else {
            console.warn('Audio data contains all zeros');
            setError('No audio signal detected');
            setIsProcessing(false);
            setIsRecording(false);
            return;
          }

          // Check audio level after normalization
          let sumAbs = 0;
          for (let i = 0; i < concatenated.length; i++) {
            sumAbs += Math.abs(concatenated[i]);
          }
          const level = sumAbs / concatenated.length;
          console.log(`Audio level after normalization: ${level}`);

          if (level < 0.001) {
            console.warn('Audio level too low after normalization:', level);
            setError('Audio level too low, please speak louder');
            setIsProcessing(false);
            setIsRecording(false);
            return;
          }
          
          // Convert to base64
          const uint8Array = new Uint8Array(concatenated.buffer);
          let base64Data = '';
          
          console.log("Converting to base64...");
          
          for (let i = 0; i < uint8Array.length; i++) {
            base64Data += String.fromCharCode(uint8Array[i]);
          }
          
          base64Data = btoa(base64Data);
          console.log("Base64 conversion complete. Length:", base64Data.length);
          
          // Strip trailing slashes and construct URL
          const baseUrl = process.env.NEXT_PUBLIC_STT_SERVER_URL?.replace(/\/+$/, '') || 'http://localhost:8000';
          const transcribeUrl = `${baseUrl}/transcribe`;
          console.log("Transcription URL:", transcribeUrl);

          // Send to STT service
          const response = await fetch(transcribeUrl, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ audio: base64Data })
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const result = await response.json();

          if (result.transcription) {
            // Check for the common "Thank you" error response
            if (result.transcription.trim() === "Thank you." || result.transcription.trim() === "Thank you") {
              console.error("Received 'Thank you' error response from STT server");
              setError("Speech recognition error: The STT server returned an error response. Please try again.");
              return;
            }
            
            // Update transcription display
            setTranscription(result.transcription);
            
            // Process the transcription to extract data
            try {
              const extractedData = await extractData(result.transcription);
              setExtractedData(extractedData);
            } catch (err) {
              setError(`Failed to extract data: ${err instanceof Error ? err.message : "Unknown error"}`);
            }
          } else {
            setError('No speech detected');
          }
        } catch (error) {
          console.error("Error processing recording:", error);
          setError('Failed to process your recording');
        } finally {
          setIsProcessing(false);
          setIsRecording(false);
          
          // Clean up
          processor.disconnect();
          source.disconnect();
          audioContext.close();
        }
      };
      
      mediaRecorder.onerror = (event) => {
        console.error("MediaRecorder error:", event);
        setError('Recording error occurred');
        setIsRecording(false);
      };
      
      console.log("Starting MediaRecorder");
      mediaRecorder.start(1000);
      console.log("MediaRecorder started");
      
      // Auto-stop after 20 seconds
      setTimeout(() => {
        if (isRecording && mediaRecorderRef.current?.state === 'recording') {
          console.log('Auto-stopping recording after 20 seconds');
          stopRecording();
        }
      }, 20000);
    } catch (error: any) {
      console.error("Error starting recording:", error);
      let errorMessage = 'Could not access microphone';
      
      if (error.name === 'NotAllowedError') {
        errorMessage = 'Microphone access denied. Please allow microphone access in your browser settings.';
      } else if (error.name === 'NotFoundError') {
        errorMessage = 'No microphone found. Please connect a microphone and try again.';
      }
      
      setError(errorMessage);
      setIsRecording(false);
    }
  };

  // Simple stopRecording function
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    // Stop and release the stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    setIsRecording(false);
  };

  const handleViewPatient = async (patientId: number) => {
    try {
      setIsLoadingPatients(true);
      const patientDetails = await getPatientDetails(patientId);
      setSelectedPatient(patientDetails);
      onOpen();
    } catch (error) {
      setError("Failed to load patient details");
      console.error(error);
    } finally {
      setIsLoadingPatients(false);
    }
  };

  const renderExtractedData = (data: ExtractedDataResponse) => {
    return (
      <div className="space-y-6">
        {/* SOAP Note - Show first and prominently when available */}
        {data.soap_note && (
          <Card className="bg-gray-800 border-none shadow-lg overflow-hidden">
            <CardBody className="p-0">
              <div className="bg-indigo-600 px-4 py-3 flex justify-between items-center">
                <h3 className="font-semibold text-white text-lg">SOAP Note</h3>
                <span className="bg-indigo-900 text-white text-xs px-2 py-1 rounded">AI Generated</span>
              </div>
              <div className="p-4 space-y-5 max-h-[70vh] overflow-y-auto">
                {/* Subjective Section */}
                <div>
                  <h4 className="text-lg font-semibold text-indigo-400 mb-2 border-b border-indigo-400 pb-1">Subjective</h4>
                  <div className="space-y-3">
                    <div className="bg-gray-700 p-3 rounded">
                      <div className="text-gray-300 text-sm mb-1 font-semibold">Chief Complaint</div>
                      <div className="text-white">{data.soap_note.SOAP.Subjective.ChiefComplaint}</div>
                    </div>
                    
                    <div className="bg-gray-700 p-3 rounded">
                      <div className="text-gray-300 text-sm mb-2 font-semibold">History of Present Illness</div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-white">
                        <div>
                          <span className="text-gray-400 font-medium">Onset: </span>
                          {data.soap_note.SOAP.Subjective.HistoryOfPresentIllness.Onset}
                        </div>
                        <div>
                          <span className="text-gray-400 font-medium">Location: </span>
                          {data.soap_note.SOAP.Subjective.HistoryOfPresentIllness.Location}
                        </div>
                        <div>
                          <span className="text-gray-400 font-medium">Duration: </span>
                          {data.soap_note.SOAP.Subjective.HistoryOfPresentIllness.Duration}
                        </div>
                        <div>
                          <span className="text-gray-400 font-medium">Characteristics: </span>
                          {data.soap_note.SOAP.Subjective.HistoryOfPresentIllness.Characteristics}
                        </div>
                        <div>
                          <span className="text-gray-400 font-medium">Aggravating Factors: </span>
                          {data.soap_note.SOAP.Subjective.HistoryOfPresentIllness.AggravatingFactors}
                        </div>
                        <div>
                          <span className="text-gray-400 font-medium">Relieving Factors: </span>
                          {data.soap_note.SOAP.Subjective.HistoryOfPresentIllness.RelievingFactors}
                        </div>
                        <div>
                          <span className="text-gray-400 font-medium">Timing: </span>
                          {data.soap_note.SOAP.Subjective.HistoryOfPresentIllness.Timing}
                        </div>
                        <div>
                          <span className="text-gray-400 font-medium">Severity: </span>
                          {data.soap_note.SOAP.Subjective.HistoryOfPresentIllness.Severity}
                        </div>
                      </div>
                    </div>

                    {data.soap_note.SOAP.Subjective.PastMedicalHistory && (
                      <div className="bg-gray-700 p-3 rounded">
                        <div className="text-gray-300 text-sm mb-1 font-semibold">Past Medical History</div>
                        <div className="text-white">{data.soap_note.SOAP.Subjective.PastMedicalHistory}</div>
                      </div>
                    )}

                    {data.soap_note.SOAP.Subjective.FamilyHistory && (
                      <div className="bg-gray-700 p-3 rounded">
                        <div className="text-gray-300 text-sm mb-1 font-semibold">Family History</div>
                        <div className="text-white">{data.soap_note.SOAP.Subjective.FamilyHistory}</div>
                      </div>
                    )}

                    {data.soap_note.SOAP.Subjective.SocialHistory && (
                      <div className="bg-gray-700 p-3 rounded">
                        <div className="text-gray-300 text-sm mb-1 font-semibold">Social History</div>
                        <div className="text-white">{data.soap_note.SOAP.Subjective.SocialHistory}</div>
                      </div>
                    )}

                    {data.soap_note.SOAP.Subjective.ReviewOfSystems && (
                      <div className="bg-gray-700 p-3 rounded">
                        <div className="text-gray-300 text-sm mb-1 font-semibold">Review of Systems</div>
                        <div className="text-white">{data.soap_note.SOAP.Subjective.ReviewOfSystems}</div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Assessment Section */}
                <div>
                  <h4 className="text-lg font-semibold text-indigo-400 mb-2 border-b border-indigo-400 pb-1">Assessment</h4>
                  <div className="space-y-3">
                    {data.soap_note.SOAP.Assessment.PrimaryDiagnosis && (
                      <div className="bg-gray-700 p-3 rounded">
                        <div className="text-gray-300 text-sm mb-1 font-semibold">Primary Diagnosis</div>
                        <div className="text-white">{data.soap_note.SOAP.Assessment.PrimaryDiagnosis}</div>
                      </div>
                    )}

                    {data.soap_note.SOAP.Assessment.DifferentialDiagnosis && (
                      <div className="bg-gray-700 p-3 rounded">
                        <div className="text-gray-300 text-sm mb-1 font-semibold">Differential Diagnosis</div>
                        <div className="text-white">{data.soap_note.SOAP.Assessment.DifferentialDiagnosis}</div>
                      </div>
                    )}

                    {data.soap_note.SOAP.Assessment.ProblemList && (
                      <div className="bg-gray-700 p-3 rounded">
                        <div className="text-gray-300 text-sm mb-1 font-semibold">Problem List</div>
                        <div className="text-white">{data.soap_note.SOAP.Assessment.ProblemList}</div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Plan Section */}
                <div>
                  <h4 className="text-lg font-semibold text-indigo-400 mb-2 border-b border-indigo-400 pb-1">Plan</h4>
                  <div className="space-y-3">
                    {data.soap_note.SOAP.Plan.TreatmentAndMedications && (
                      <div className="bg-gray-700 p-3 rounded">
                        <div className="text-gray-300 text-sm mb-1 font-semibold">Treatment and Medications</div>
                        <div className="text-white">{data.soap_note.SOAP.Plan.TreatmentAndMedications}</div>
                      </div>
                    )}

                    {data.soap_note.SOAP.Plan.FurtherTestingOrImaging && (
                      <div className="bg-gray-700 p-3 rounded">
                        <div className="text-gray-300 text-sm mb-1 font-semibold">Further Testing or Imaging</div>
                        <div className="text-white">{data.soap_note.SOAP.Plan.FurtherTestingOrImaging}</div>
                      </div>
                    )}

                    {data.soap_note.SOAP.Plan.PatientEducation && (
                      <div className="bg-gray-700 p-3 rounded">
                        <div className="text-gray-300 text-sm mb-1 font-semibold">Patient Education</div>
                        <div className="text-white">{data.soap_note.SOAP.Plan.PatientEducation}</div>
                      </div>
                    )}

                    {data.soap_note.SOAP.Plan.FollowUp && (
                      <div className="bg-gray-700 p-3 rounded">
                        <div className="text-gray-300 text-sm mb-1 font-semibold">Follow Up</div>
                        <div className="text-white">{data.soap_note.SOAP.Plan.FollowUp}</div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </CardBody>
          </Card>
        )}

        {/* Extracted data cards in a grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="bg-blue-700 border-none shadow-lg overflow-hidden">
            <CardBody className="p-0">
              <div className="bg-blue-600 px-4 py-3">
                <h3 className="font-semibold text-white">Patient Information</h3>
              </div>
              <div className="p-4 text-white">
                {data.name ? (
                  <dl className="space-y-2">
                    <div>
                      <dt className="text-gray-300 text-sm">Name</dt>
                      <dd>{data.name}</dd>
                    </div>
                    {data.dob && (
                      <div>
                        <dt className="text-gray-300 text-sm">Date of Birth</dt>
                        <dd>{data.dob}</dd>
                      </div>
                    )}
                    {data.address && (
                      <div>
                        <dt className="text-gray-300 text-sm">Address</dt>
                        <dd>{data.address}</dd>
                      </div>
                    )}
                    {data.phone && (
                      <div>
                        <dt className="text-gray-300 text-sm">Phone</dt>
                        <dd>{data.phone}</dd>
                      </div>
                    )}
                    {data.email && (
                      <div>
                        <dt className="text-gray-300 text-sm">Email</dt>
                        <dd>{data.email}</dd>
                      </div>
                    )}
                    {data.insurance && (
                      <div>
                        <dt className="text-gray-300 text-sm">Insurance</dt>
                        <dd>{data.insurance}</dd>
                      </div>
                    )}
                  </dl>
                ) : (
                  <div className="text-gray-300">No patient information extracted</div>
                )}
              </div>
            </CardBody>
          </Card>

          <Card className="bg-purple-800 border-none shadow-lg overflow-hidden">
            <CardBody className="p-0">
              <div className="bg-purple-700 px-4 py-3">
                <h3 className="font-semibold text-white">Medical Information</h3>
              </div>
              <div className="p-4 text-white">
                {(data.condition || data.symptoms?.length) ? (
                  <dl className="space-y-2">
                    {data.condition && (
                      <div>
                        <dt className="text-gray-300 text-sm">Condition</dt>
                        <dd>{data.condition}</dd>
                      </div>
                    )}
                    {data.symptoms && data.symptoms.length > 0 && (
                      <div>
                        <dt className="text-gray-300 text-sm">Symptoms</dt>
                        <dd>
                          <ul className="list-disc pl-5 space-y-1">
                            {data.symptoms.map((symptom, index) => (
                              <li key={index}>{symptom}</li>
                            ))}
                          </ul>
                        </dd>
                      </div>
                    )}
                  </dl>
                ) : (
                  <div className="text-gray-300">No medical information extracted</div>
                )}
              </div>
            </CardBody>
          </Card>

          <Card className="bg-green-700 border-none shadow-lg overflow-hidden">
            <CardBody className="p-0">
              <div className="bg-green-600 px-4 py-3">
                <h3 className="font-semibold text-white">Visit Information</h3>
              </div>
              <div className="p-4 text-white">
                {(data.reason_for_visit || data.appointment_details) ? (
                  <dl className="space-y-2">
                    {data.reason_for_visit && (
                      <div>
                        <dt className="text-gray-300 text-sm">Reason for Visit</dt>
                        <dd>{data.reason_for_visit}</dd>
                      </div>
                    )}
                    {data.appointment_details && (
                      <>
                        {data.appointment_details.type && (
                          <div>
                            <dt className="text-gray-300 text-sm">Type</dt>
                            <dd>{data.appointment_details.type}</dd>
                          </div>
                        )}
                        {data.appointment_details.time && (
                          <div>
                            <dt className="text-gray-300 text-sm">Time</dt>
                            <dd>{data.appointment_details.time}</dd>
                          </div>
                        )}
                        {data.appointment_details.doctor && (
                          <div>
                            <dt className="text-gray-300 text-sm">Doctor</dt>
                            <dd>{data.appointment_details.doctor}</dd>
                          </div>
                        )}
                        {data.appointment_details.scheduled_date && (
                          <div>
                            <dt className="text-gray-300 text-sm">Date</dt>
                            <dd>{data.appointment_details.scheduled_date}</dd>
                          </div>
                        )}
                      </>
                    )}
                  </dl>
                ) : (
                  <div className="text-gray-300">No visit information extracted</div>
                )}
              </div>
            </CardBody>
          </Card>
        </div>

        {/* File Information */}
        {data.files && (
          <Card className="bg-gray-800 border-none shadow-lg overflow-hidden">
            <CardBody className="p-0">
              <div className="bg-yellow-600 px-4 py-3">
                <h3 className="font-semibold text-white">Saved Files</h3>
              </div>
              <div className="p-4 text-white">
                <ul className="list-disc pl-5 space-y-1">
                  <li>Raw YAML: {data.files.raw_yaml}</li>
                  <li>Processed YAML: {data.files.processed_yaml}</li>
                  {data.files.soap_note && (
                    <li>SOAP Note: {data.files.soap_note}</li>
                  )}
                </ul>
              </div>
            </CardBody>
          </Card>
        )}
      </div>
    );
  };

  const renderPatientsList = () => {
    return (
      <Card className="bg-gray-800 border-none shadow-lg overflow-hidden">
        <CardBody className="p-0">
          <div className="bg-blue-600 px-4 py-3">
            <h3 className="font-semibold text-white">Patients</h3>
          </div>
          <div className="p-4">
            {isLoadingPatients ? (
              <div className="flex items-center justify-center py-8">
                <div className="border-t-4 border-blue-500 border-solid rounded-full w-10 h-10 animate-spin"></div>
                <span className="ml-3 text-white">Loading patients...</span>
              </div>
            ) : patients.length > 0 ? (
              <Table aria-label="Patients Table">
                <TableHeader>
                  <TableColumn>ID</TableColumn>
                  <TableColumn>Name</TableColumn>
                  <TableColumn>Date of Birth</TableColumn>
                  <TableColumn>Phone</TableColumn>
                  <TableColumn>Actions</TableColumn>
                </TableHeader>
                <TableBody>
                  {patients.map((patient) => (
                    <TableRow key={patient.patient_id}>
                      <TableCell>{patient.patient_id}</TableCell>
                      <TableCell>{patient.name}</TableCell>
                      <TableCell>{patient.date_of_birth || patient.dob}</TableCell>
                      <TableCell>{patient.phone}</TableCell>
                      <TableCell>
                        <Button 
                          size="sm"
                          onClick={() => handleViewPatient(patient.patient_id)}
                          color="primary"
                        >
                          View Details
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <div className="text-center py-8 text-gray-400">
                No patients found
              </div>
            )}
          </div>
        </CardBody>
      </Card>
    );
  };

  const renderPatientDetails = () => {
    if (!selectedPatient) return null;

    // Get patient info either from direct properties or nested patient object
    const patientInfo = selectedPatient.patient || selectedPatient;
    const appointments = selectedPatient.appointments || selectedPatient.visits || [];
    const medicalConditions = selectedPatient.medical_conditions || [];
    
    return (
      <Modal isOpen={isOpen} onClose={onClose} size="lg">
        <ModalContent className="bg-gray-800 text-white">
          <ModalHeader className="border-b border-gray-700">
            <h3 className="text-xl font-semibold">{patientInfo.name}</h3>
          </ModalHeader>
          <ModalBody>
            <div className="space-y-4">
              <div>
                <h4 className="text-lg font-semibold text-blue-400">Patient Information</h4>
                <div className="grid grid-cols-2 gap-2 mt-2">
                  <div className="text-gray-400">Date of Birth</div>
                  <div>{patientInfo.date_of_birth || patientInfo.dob}</div>
                  <div className="text-gray-400">Address</div>
                  <div>{patientInfo.address || "Not provided"}</div>
                  <div className="text-gray-400">Phone</div>
                  <div>{patientInfo.phone}</div>
                  <div className="text-gray-400">Email</div>
                  <div>{patientInfo.email || "Not provided"}</div>
                  <div className="text-gray-400">Insurance</div>
                  <div>{patientInfo.insurance || "Not provided"}</div>
                </div>
              </div>
              
              <div>
                <h4 className="text-lg font-semibold text-purple-400">Medical Information</h4>
                {selectedPatient.medical_history ? (
                  <div className="mt-2">
                    <div className="text-gray-400 mb-1">Medical History</div>
                    <div className="bg-gray-700 p-3 rounded">{selectedPatient.medical_history}</div>
                  </div>
                ) : medicalConditions.length > 0 ? (
                  <div className="space-y-3 mt-2">
                    {medicalConditions.map((condition, idx) => (
                      <div key={idx} className="bg-gray-700 p-3 rounded">
                        <div className="font-semibold text-purple-400">{condition.name}</div>
                        {condition.symptoms.length > 0 && (
                          <div className="mt-2">
                            <div className="text-gray-400 mb-1">Symptoms</div>
                            <ul className="list-disc ml-5">
                              {condition.symptoms.map((symptom, i) => (
                                <li key={i} className="text-white">
                                  {symptom.name}
                                  {symptom.severity && <span className="text-yellow-400 ml-2">({symptom.severity})</span>}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-gray-500 mt-2">No medical information available</div>
                )}
                
                {selectedPatient.medications && selectedPatient.medications.length > 0 && (
                  <div className="mt-3">
                    <div className="text-gray-400 mb-1">Medications</div>
                    <div className="space-y-2">
                      {selectedPatient.medications.map((med: any, i: number) => (
                        <div key={i} className="bg-gray-700 p-2 rounded">
                          <div className="text-purple-400">{med.name}</div>
                          <div className="grid grid-cols-2 gap-2">
                            {med.dosage && (
                              <>
                                <div className="text-gray-400">Dosage</div>
                                <div>{med.dosage}</div>
                              </>
                            )}
                            {med.frequency && (
                              <>
                                <div className="text-gray-400">Frequency</div>
                                <div>{med.frequency}</div>
                              </>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {selectedPatient.allergies && selectedPatient.allergies.length > 0 && (
                  <div className="mt-3">
                    <div className="text-gray-400 mb-1">Allergies</div>
                    <div className="bg-gray-700 p-2 rounded">
                      <ul className="list-disc ml-4">
                        {selectedPatient.allergies.map((allergy: string, i: number) => (
                          <li key={i}>{allergy}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}
              </div>
              
              {(appointments.length > 0) && (
                <div>
                  <h4 className="text-lg font-semibold text-green-400">Appointments</h4>
                  <div className="space-y-3 mt-2">
                    {appointments.map((appointment, i) => (
                      <div key={i} className="bg-gray-700 p-3 rounded">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-green-400">{appointment.datetime || appointment.date}</span>
                          <span className="bg-blue-500 px-2 py-1 rounded text-xs">
                            {appointment.status || "Scheduled"}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <div className="text-gray-400">Type</div>
                          <div>{appointment.appointment_type || "Regular"}</div>
                          <div className="text-gray-400">Doctor</div>
                          <div>{appointment.doctor_name || appointment.doctor || "Not assigned"}</div>
                          {appointment.notes && (
                            <>
                              <div className="text-gray-400">Notes</div>
                              <div>{appointment.notes}</div>
                            </>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </ModalBody>
          <ModalFooter>
            <Button onClick={onClose} color="primary">
              Close
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    );
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === 'audio/wav') {
      setSelectedFile(file);
      setError(null);
    } else {
      setSelectedFile(null);
      setError('Please select a valid WAV file');
    }
  };

  const processUploadedFile = async () => {
    if (!selectedFile) {
      setError('No file selected');
      return;
    }

    setIsProcessingUpload(true);
    setError(null);

    try {
      // Convert file to base64
      const base64Data = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsArrayBuffer(selectedFile);
        reader.onload = () => {
          if (!reader.result) {
            reject(new Error('Failed to read file'));
            return;
          }
          
          // Convert ArrayBuffer to base64
          const bytes = new Uint8Array(reader.result as ArrayBuffer);
          let binary = '';
          for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
          }
          const base64 = btoa(binary);
          resolve(base64);
        };
        reader.onerror = () => reject(reader.error);
      });

      // Use the extractFromAudio function directly
      try {
        const extractedData = await extractFromAudio(base64Data);
        setTranscription(extractedData.transcription || "");
        setExtractedData(extractedData);
        
        // If a patient was created, refresh the patients list
        if (extractedData.patient_id) {
          loadPatients();
        }
      } catch (err) {
        if (err instanceof Error && err.message.includes('transcription')) {
          // This is a transcription-specific error
          setError(`Transcription error: ${err.message}`);
        } else {
          setError(`Failed to process audio: ${err instanceof Error ? err.message : "Unknown error"}`);
        }
      }
    } catch (error) {
      console.error('Error processing uploaded file:', error);
      setError(`Failed to process the audio file: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessingUpload(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex flex-col space-y-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <h1 className="text-3xl font-bold text-white">Data Extractor</h1>
          <Tabs 
            selectedKey={activeTab} 
            onSelectionChange={key => setActiveTab(key as string)}
            color="primary"
            variant="underlined"
            classNames={{
              tabList: "bg-gray-800 rounded-lg p-1",
              cursor: "bg-blue-600",
              tab: "text-white",
            }}
          >
            <Tab key="extract" title="Extract Data" />
            <Tab key="patients" title="View Patients" />
          </Tabs>
        </div>

        {error && (
          <div className="bg-red-900/50 border border-red-500 text-red-100 px-4 py-3 rounded">
            {error}
          </div>
        )}

        {activeTab === "extract" ? (
          <>
            <Card className="bg-gray-800 border-none shadow-lg overflow-hidden">
              <CardBody>
                <h2 className="text-xl font-semibold text-white mb-4">Voice Input</h2>
                
                <div className="mb-4">
                  <button
                    onClick={handleExtractFromAudio}
                    className={`rounded-lg px-6 py-3 flex items-center gap-2 ${
                      isRecording 
                        ? 'bg-red-600 hover:bg-red-700' 
                        : 'bg-blue-600 hover:bg-blue-700'
                    } text-white disabled:opacity-50 disabled:cursor-not-allowed`}
                    disabled={isProcessing}
                  >
                    {isRecording ? (
                      <>
                        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                        </svg>
                        Stop Recording
                      </>
                    ) : (
                      <>
                        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                        </svg>
                        Start Recording
                      </>
                    )}
                  </button>
                </div>
                
                {isProcessing && (
                  <div className="flex items-center justify-center my-4">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
                    <span className="ml-3 text-white">Processing audio...</span>
                  </div>
                )}
                
                <h2 className="text-xl font-semibold text-white mb-2 mt-6">Transcription</h2>
                <div className="mb-2 flex justify-between items-center">
                  <div className="text-gray-400 text-sm">
                    {transcription ? `${transcription.length} characters` : "No transcription yet"}
                  </div>
                  {transcription && (
                    <Button
                      size="sm"
                      color="danger"
                      variant="flat"
                      onClick={() => setTranscription("")}
                    >
                      Clear
                    </Button>
                  )}
                </div>
                <Textarea
                  value={transcription}
                  onChange={(e) => setTranscription(e.target.value)}
                  placeholder="Your transcription will appear here..."
                  className="w-full"
                  rows={8}
                />
                
                <div className="flex flex-col md:flex-row gap-3 mt-4">
                  <Button
                    onClick={handleExtractData}
                    color="primary"
                    isLoading={isProcessing}
                    isDisabled={!transcription.trim() || isProcessing}
                  >
                    Extract Data from Text
                  </Button>
                </div>
              </CardBody>
            </Card>
            
            <div className="mb-6">
              <h3 className="font-medium text-white mb-2">Or upload a WAV file:</h3>
              <div className="flex flex-col sm:flex-row gap-3">
                <input
                  type="file"
                  accept="audio/wav"
                  onChange={handleFileUpload}
                  className="bg-gray-700 rounded-lg p-2 text-white"
                />
                <Button
                  className="bg-purple-600 hover:bg-purple-700 text-white"
                  onClick={processUploadedFile}
                  disabled={!selectedFile || isProcessingUpload}
                >
                  {isProcessingUpload ? (
                    <>
                      <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></span>
                      Processing...
                    </>
                  ) : (
                    'Process WAV File'
                  )}
                </Button>
              </div>
              {selectedFile && (
                <p className="text-sm text-gray-300 mt-2">
                  Selected file: {selectedFile.name} ({Math.round(selectedFile.size / 1024)} KB)
                </p>
              )}
            </div>
            
            {extractedData && (
              <Card className="bg-gray-800 border-none shadow-lg">
                <CardBody>
                  <h2 className="text-xl font-semibold text-white mb-6">Extracted Data</h2>
                  {renderExtractedData(extractedData)}
                </CardBody>
              </Card>
            )}
          </>
        ) : (
          renderPatientsList()
        )}
      </div>
      
      {renderPatientDetails()}
    </div>
  );
}