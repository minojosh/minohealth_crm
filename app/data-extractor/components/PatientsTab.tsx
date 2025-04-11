import React, { useState, useEffect, useCallback } from 'react';
import { getAllPatients, getPatientDetails, PatientDetails, PatientDetailsResponse } from '../api';
import { Table, TableHeader, TableColumn, TableBody, TableRow, TableCell } from "@heroui/table";
import { Button } from "@heroui/button";
import { Modal, ModalContent, ModalHeader, ModalBody, ModalFooter, useDisclosure } from "@heroui/modal";
import { Progress } from "@heroui/progress";
import { Card, CardBody, CardHeader } from "@heroui/card";
import { Listbox, ListboxItem } from "@heroui/listbox";
import { Chip } from "@heroui/chip";

// Reusable Dl renderer from ExtractedDataDisplay (consider moving to a shared utils file)
const RenderSimpleDl: React.FC<{ items: Record<string, string | undefined | null>, className?: string }> = ({ items, className }) => (
    <dl className={`grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-1 text-sm ${className}`}>
        {Object.entries(items).map(([key, value]) => value ? (
            <React.Fragment key={key}>
                <dt className="font-semibold text-gray-400">{key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}</dt>
                <dd className="text-gray-200 md:col-start-2">{value}</dd>
            </React.Fragment>
        ) : null)}
    </dl>
);

const PatientsTableComponent: React.FC<{ patients: PatientDetails[], onSelectPatient: (id: number) => void, isLoading: boolean }> = ({ patients, onSelectPatient, isLoading }) => {
     if (isLoading) {
         return <Progress size="md" isIndeterminate aria-label="Loading patients..." className="max-w-full" />;
     }

     return (
         <Table aria-label="Patients List" removeWrapper className="bg-transparent shadow-none">
             <TableHeader>
                 <TableColumn>ID</TableColumn>
                 <TableColumn>Name</TableColumn>
                 <TableColumn>Date of Birth</TableColumn>
                 <TableColumn>Phone</TableColumn>
                 <TableColumn>Actions</TableColumn>
             </TableHeader>
             <TableBody items={patients} emptyContent="No patients found.">
                 {(patient) => (
                     <TableRow key={patient.patient_id}>
                         <TableCell>{patient.patient_id}</TableCell>
                         <TableCell>{patient.name}</TableCell>
                         <TableCell>{patient.date_of_birth || patient.dob || 'N/A'}</TableCell>
                         <TableCell>{patient.phone}</TableCell>
                         <TableCell>
                             <Button
                                size="sm"
                                color="primary"
                                variant="flat"
                                onPress={() => onSelectPatient(patient.patient_id)}
                             >
                                 View Details
                             </Button>
                         </TableCell>
                     </TableRow>
                 )}
             </TableBody>
         </Table>
     );
};

const PatientDetailsModalComponent: React.FC<{ patient: PatientDetailsResponse | null, isOpen: boolean, onClose: () => void }> = ({ patient, isOpen, onClose }) => {
    if (!patient) return null;

    // Extract details for easier access
    const patientInfo = patient.patient || patient;
    const appointments = patient.appointments || patient.visits || [];
    const medicalConditions = patient.medical_conditions || [];
    const medications = patient.medications || [];
    const allergies = patient.allergies || [];

    return (
        <Modal isOpen={isOpen} onClose={onClose} size="3xl" backdrop="blur" scrollBehavior="inside">
            <ModalContent className="bg-gray-800 text-white">
                <ModalHeader className="border-b border-gray-700">
                    <h3 className="text-xl font-semibold">Patient Details: {patientInfo.name}</h3>
                </ModalHeader>
                <ModalBody className="py-4 px-6 space-y-6">
                     {/* Patient Demographics */}
                    <Card shadow="sm" className="bg-gray-700">
                         <CardHeader className="text-blue-300 font-semibold text-base">Patient Information</CardHeader>
                         <CardBody>
                             <RenderSimpleDl items={{
                                 "Date of Birth": patientInfo.date_of_birth || patientInfo.dob,
                                 Address: patientInfo.address,
                                 Phone: patientInfo.phone,
                                 Email: patientInfo.email,
                                 Insurance: patientInfo.insurance
                             }} />
                         </CardBody>
                    </Card>

                    {/* Medical Info */} 
                    <Card shadow="sm" className="bg-gray-700">
                        <CardHeader className="text-purple-300 font-semibold text-base">Medical Information</CardHeader>
                        <CardBody className="space-y-4">
                             {patient.medical_history && (
                                 <div>
                                     <h4 className="font-semibold text-gray-400 mb-1">Medical History</h4>
                                     <p className="text-sm text-gray-200 bg-gray-600 p-2 rounded">{patient.medical_history}</p>
                                 </div>
                             )}
                             {medicalConditions.length > 0 && (
                                <div>
                                     <h4 className="font-semibold text-gray-400 mb-2">Conditions</h4>
                                     {medicalConditions.map((condition, idx) => (
                                         <div key={idx} className="mb-2 p-2 bg-gray-600 rounded">
                                             <p className="font-medium text-purple-300">{condition.name}</p>
                                             {condition.symptoms.length > 0 && (
                                                <div className="mt-1">
                                                     <p className="text-xs text-gray-400 mb-1">Symptoms:</p>
                                                     <ul className="list-disc ml-5 text-xs text-gray-200">
                                                         {condition.symptoms.map((symptom, i) => (
                                                            <li key={i}>{symptom.name} {symptom.severity && `(${symptom.severity})`}</li>
                                                         ))}
                                                     </ul>
                                                </div>
                                             )}
                                         </div>
                                     ))}
                                </div>
                             )}
                            {medications.length > 0 && (
                                <div>
                                    <h4 className="font-semibold text-gray-400 mb-1">Medications</h4>
                                    <Listbox items={medications} aria-label="Medications" variant="flat" className="bg-gray-600 rounded p-2">
                                         {(med) => (
                                             <ListboxItem key={med.name} textValue={med.name} className="text-white">
                                                 <div className="flex justify-between items-center">
                                                     <span>{med.name}</span>
                                                     <span className="text-xs text-gray-400">
                                                         {med.dosage}{med.frequency && ` - ${med.frequency}`}
                                                     </span>
                                                 </div>
                                             </ListboxItem>
                                         )}
                                     </Listbox>
                                </div>
                             )}
                             {allergies.length > 0 && (
                                <div>
                                     <h4 className="font-semibold text-gray-400 mb-1">Allergies</h4>
                                     <div className="flex flex-wrap gap-1">
                                        {allergies.map((allergy, i) => <Chip key={i} color="warning" size="sm" variant="flat">{allergy}</Chip>)}
                                     </div>
                                </div>
                             )}
                             {!(patient.medical_history || medicalConditions.length || medications.length || allergies.length) && 
                                <p className="text-sm text-gray-400 italic">No medical information recorded.</p>}
                         </CardBody>
                    </Card>

                    {/* Appointments */} 
                    {appointments.length > 0 && (
                        <Card shadow="sm" className="bg-gray-700">
                             <CardHeader className="text-green-300 font-semibold text-base">Appointments</CardHeader>
                             <CardBody className="space-y-3">
                                 {appointments.map((appt, i) => (
                                    <div key={i} className="bg-gray-600 p-3 rounded">
                                         <div className="flex justify-between items-center mb-2">
                                             <span className="text-green-300 text-sm font-medium">{appt.datetime || appt.scheduled_date || appt.date}</span>
                                             <Chip size="sm" color={appt.status === 'Completed' ? 'success' : 'primary'} variant="flat">{appt.status || "Scheduled"}</Chip>
                                         </div>
                                         <RenderSimpleDl className="text-xs" items={{
                                             Type: appt.appointment_type,
                                             Doctor: appt.doctor_name || appt.doctor,
                                             Notes: appt.notes
                                         }} />
                                    </div>
                                 ))}
                             </CardBody>
                        </Card>
                    )}

                </ModalBody>
                <ModalFooter>
                    <Button color="danger" variant="light" onPress={onClose}>
                         Close
                    </Button>
                </ModalFooter>
            </ModalContent>
        </Modal>
    );
};

const PatientsTab: React.FC = () => {
  const [patients, setPatients] = useState<PatientDetails[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<PatientDetailsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { isOpen, onOpen, onClose } = useDisclosure();

  const loadPatients = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await getAllPatients();
      setPatients(response.patients);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load patients');
    } finally {
      setIsLoading(false);
    }
  }, []); // Add useCallback with empty dependency array

  const viewPatient = useCallback(async (id: number) => {
     // Don't set main loading true, modal should have its own indicator if needed
     setError(null);
     try {
        const details = await getPatientDetails(id);
        setSelectedPatient(details);
        onOpen(); // Open modal after fetching data
     } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load patient details');
     } finally {
        // setIsLoading(false); // Remove this if not setting true
     }
  }, [onOpen]); // Add onOpen to dependency array

  // Load patients on mount
  useEffect(() => {
    loadPatients();
  }, [loadPatients]); // Use useCallback function in dependency array

  return (
     <Card className="bg-gray-800 border-none shadow-lg">
         <CardBody className="p-6 space-y-4">
             <div className="flex justify-between items-center">
                 <h2 className="text-xl font-semibold text-white">Patients List</h2>
                 <Button 
                    isIconOnly 
                    size="sm" 
                    variant="flat" 
                    onPress={loadPatients} 
                    isDisabled={isLoading}
                    aria-label="Refresh List"
                 >
                     <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                         <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99" />
                     </svg>
                 </Button>
             </div>

            {error && <p className="text-red-500 text-sm">Error: {error}</p>}
            
            <PatientsTableComponent
                patients={patients}
                onSelectPatient={viewPatient}
                isLoading={isLoading}
            />
            
            <PatientDetailsModalComponent
                patient={selectedPatient}
                isOpen={isOpen}
                onClose={onClose}
            />
         </CardBody>
     </Card>
  );
};

export default PatientsTab; 