import React from 'react';
import { ExtractedDataResponse, SOAPNote } from '../api'; // Adjust path if needed
import { Card, CardBody, CardHeader } from "@heroui/card";
import { Listbox, ListboxItem } from "@heroui/listbox"; // For lists
import { Accordion, AccordionItem } from "@heroui/accordion"; // Good for SOAP notes
import { Chip } from "@heroui/chip"; // For tags/status

interface ExtractedDataDisplayProps {
    data: ExtractedDataResponse | null;
}

const RenderSimpleDl: React.FC<{ items: Record<string, string | undefined | null> }> = ({ items }) => (
    <dl className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-1 text-sm">
        {Object.entries(items).map(([key, value]) => value ? (
            <React.Fragment key={key}>
                <dt className="font-semibold text-gray-400">{key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}</dt>
                <dd className="text-gray-200 md:col-start-2">{value}</dd>
            </React.Fragment>
        ) : null)}
    </dl>
);

const RenderList: React.FC<{ title: string, items: string[] | undefined | null }> = ({ title, items }) => items && items.length > 0 ? (
    <div>
        <h4 className="font-semibold text-gray-400 mb-1">{title}</h4>
        <ul className="list-disc pl-5 text-sm text-gray-200 space-y-1">
            {items.map((item, index) => <li key={index}>{item}</li>)}
        </ul>
    </div>
) : null;

const RenderSoapNote: React.FC<{ soap: SOAPNote['SOAP'] }> = ({ soap }) => {
    return (
        <Accordion variant="splitted" defaultExpandedKeys={["subjective"]} itemClasses={{ base: "bg-gray-700 text-white", title: "text-base font-semibold text-indigo-400" }}>
            <AccordionItem key="subjective" aria-label="Subjective" title="Subjective">
                <div className="space-y-3 p-2">
                    <RenderSimpleDl items={{ "Chief Complaint": soap.Subjective.ChiefComplaint }} />
                     <Card shadow="none" className="bg-gray-600 p-3">
                         <CardHeader className="text-sm font-semibold text-gray-300 pb-1">History of Present Illness</CardHeader>
                         <CardBody className="pt-1">
                            <RenderSimpleDl items={soap.Subjective.HistoryOfPresentIllness} />
                         </CardBody>
                     </Card>
                     {soap.Subjective.PastMedicalHistory && <RenderSimpleDl items={{ "Past Medical History": soap.Subjective.PastMedicalHistory}} />} 
                     {soap.Subjective.FamilyHistory && <RenderSimpleDl items={{ "Family History": soap.Subjective.FamilyHistory}} />} 
                     {soap.Subjective.SocialHistory && <RenderSimpleDl items={{ "Social History": soap.Subjective.SocialHistory}} />} 
                     {soap.Subjective.ReviewOfSystems && <RenderSimpleDl items={{ "Review of Systems": soap.Subjective.ReviewOfSystems}} />} 
                 </div>
            </AccordionItem>
            <AccordionItem key="assessment" aria-label="Assessment" title="Assessment">
                 <div className="space-y-3 p-2">
                    {soap.Assessment.PrimaryDiagnosis && <RenderSimpleDl items={{ "Primary Diagnosis": soap.Assessment.PrimaryDiagnosis}} />} 
                    {soap.Assessment.DifferentialDiagnosis && <RenderSimpleDl items={{ "Differential Diagnosis": soap.Assessment.DifferentialDiagnosis}} />} 
                    {soap.Assessment.ProblemList && <RenderSimpleDl items={{ "Problem List": soap.Assessment.ProblemList}} />} 
                 </div>
            </AccordionItem>
            <AccordionItem key="plan" aria-label="Plan" title="Plan">
                 <div className="space-y-3 p-2">
                    {soap.Plan.TreatmentAndMedications && <RenderSimpleDl items={{ "Treatment/Medications": soap.Plan.TreatmentAndMedications}} />} 
                    {soap.Plan.FurtherTestingOrImaging && <RenderSimpleDl items={{ "Further Testing/Imaging": soap.Plan.FurtherTestingOrImaging}} />} 
                    {soap.Plan.PatientEducation && <RenderSimpleDl items={{ "Patient Education": soap.Plan.PatientEducation}} />} 
                    {soap.Plan.FollowUp && <RenderSimpleDl items={{ "Follow Up": soap.Plan.FollowUp}} />} 
                 </div>
            </AccordionItem>
        </Accordion>
    );
};

const ExtractedDataDisplay: React.FC<ExtractedDataDisplayProps> = ({ data }) => {
    if (!data) return null;

    return (
        <div className="mt-6 space-y-6">
             <h2 className="text-xl font-semibold text-white mb-4">Extracted Data</h2>
            
            {/* SOAP Note Card */} 
            {data.soap_note && (
                 <Card className="bg-gray-800 border-none shadow-lg overflow-hidden">
                     <CardHeader className="bg-indigo-700 px-4 py-3 flex justify-between items-center">
                         <h3 className="font-semibold text-white text-lg">SOAP Note</h3>
                         <Chip size="sm" color="secondary" variant="flat">AI Generated</Chip>
                     </CardHeader>
                     <CardBody className="p-0">
                         <RenderSoapNote soap={data.soap_note.SOAP} />
                     </CardBody>
                 </Card>
            )}

            {/* Other data in a grid */} 
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                 <Card shadow="sm" className="bg-blue-900/50 border border-blue-700">
                     <CardHeader className="text-blue-300 font-semibold">Patient Info</CardHeader>
                     <CardBody>
                         <RenderSimpleDl items={{
                             Name: data.name,
                             DOB: data.dob,
                             Address: data.address,
                             Phone: data.phone,
                             Email: data.email,
                             Insurance: data.insurance
                         }} />
                         {!data.name && <p className="text-sm text-gray-400 italic">No patient info extracted.</p>}
                     </CardBody>
                 </Card>

                 <Card shadow="sm" className="bg-purple-900/50 border border-purple-700">
                     <CardHeader className="text-purple-300 font-semibold">Medical Info</CardHeader>
                     <CardBody className="space-y-2">
                         <RenderSimpleDl items={{ Condition: data.condition }} />
                         <RenderList title="Symptoms" items={data.symptoms} />
                         {!(data.condition || data.symptoms?.length) && <p className="text-sm text-gray-400 italic">No medical info extracted.</p>}
                     </CardBody>
                 </Card>

                 <Card shadow="sm" className="bg-green-900/50 border border-green-700">
                     <CardHeader className="text-green-300 font-semibold">Visit Info</CardHeader>
                     <CardBody>
                         <RenderSimpleDl items={{
                             ReasonForVisit: data.reason_for_visit,
                             Type: data.appointment_details?.type,
                             Time: data.appointment_details?.time,
                             Doctor: data.appointment_details?.doctor,
                             Date: data.appointment_details?.scheduled_date
                         }} />
                         {!(data.reason_for_visit || data.appointment_details) && <p className="text-sm text-gray-400 italic">No visit info extracted.</p>}
                     </CardBody>
                 </Card>
            </div>

            {/* Files Card */} 
            {data.files && (
                 <Card shadow="sm" className="bg-yellow-900/50 border border-yellow-700">
                     <CardHeader className="text-yellow-300 font-semibold">Files</CardHeader>
                     <CardBody>
                        <RenderSimpleDl items={{
                             RawYaml: data.files.raw_yaml,
                             ProcessedYaml: data.files.processed_yaml,
                             SoapNoteFile: data.files.soap_note
                        }} />
                     </CardBody>
                 </Card>
            )}
        </div>
    );
};

export default ExtractedDataDisplay; 