"use client";

import { useState, useEffect } from 'react';
import { Button } from '@heroui/button';
import { Card, CardBody } from '@heroui/card';
import { Divider } from '@heroui/divider';
import { Input } from '@heroui/input';
import { Select, SelectItem } from '@heroui/select';
import { Table, TableHeader, TableBody, TableColumn, TableRow, TableCell } from '@heroui/table';
import { ReminderResponse } from './types';
import { AppointmentConversation } from '../../components/voice/AppointmentConversation';
import { MedicationConversation } from '../../components/voice/MedicationConversation';
import { appointmentManagerApi } from './api';
import { 
  ClockIcon, 
  BellIcon, 
  CalendarIcon, 
  InformationCircleIcon, 
  CheckCircleIcon,
  ArrowPathIcon,
  SpeakerWaveIcon
} from '@heroicons/react/24/outline';

enum FlowStep {
  SEARCH,
  CONVERSATION,
  RESULTS
}

export default function AppointmentManager() {
  const [type, setType] = useState<'appointment' | 'medication'>('appointment');
  const [daysAhead, setDaysAhead] = useState<number>(1);
  const [hoursAhead, setHoursAhead] = useState<number>(1);
  const [loading, setLoading] = useState(false);
  const [reminders, setReminders] = useState<ReminderResponse[]>([]);
  const [selectedReminder, setSelectedReminder] = useState<ReminderResponse | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [appointmentResult, setAppointmentResult] = useState<any>(null);
  const [searchPerformed, setSearchPerformed] = useState(false);
  const [currentStep, setCurrentStep] = useState<FlowStep>(FlowStep.SEARCH);
  
  // Process a single medication or appointment at a time
  const handleSchedule = async () => {
    setLoading(true);
    setError(null);
    setSearchPerformed(false);
    setReminders([]);
    
    try {
      const response = await appointmentManagerApi.startScheduler({
        type,
        days_ahead: daysAhead,
        hours_ahead: hoursAhead,
      });
      
      setReminders(response);
      setSearchPerformed(true);
      
      // Show feedback about the search results
      if (response.length === 0) {
        setError(`No ${type} reminders found within the specified timeframe.`);
      } else {
        // Automatically start conversation with the first reminder
        setSelectedReminder(response[0]);
        setCurrentStep(FlowStep.CONVERSATION);
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to schedule appointment');
    } finally {
      setLoading(false);
    }
  };
  
  const handleStartConversation = (reminder: ReminderResponse) => {
    setSelectedReminder(reminder);
    setCurrentStep(FlowStep.CONVERSATION);
  };

  const handleCompleteConversation = (result: any) => {
    if (!result) {
      setError("No result was produced. The conversation may have failed.");
      setCurrentStep(FlowStep.SEARCH);
      return;
    }
    
    setAppointmentResult(result);
    setCurrentStep(FlowStep.RESULTS);
  };
  
  // Format the reminder type for display
  const formatReminderType = (type: string) => {
    return type.charAt(0).toUpperCase() + type.slice(1);
  };
  
  // Format details for display based on reminder type
  const formatReminderDetails = (reminder: ReminderResponse) => {
    if (reminder.message_type === 'appointment') {
      return `Dr. ${reminder.details.doctor_name} - ${new Date(reminder.details.datetime).toLocaleString()}`;
    } else if (reminder.message_type === 'medication') {
      return `${reminder.details.medication_name} - ${reminder.details.dosage} ${reminder.details.frequency || ''}`;
    }
    return 'Details not available';
  };
  
  // Get status badge based on reminder type
  const getStatusBadge = (reminder: ReminderResponse) => {
    const baseClasses = "px-2 py-1 rounded-full text-xs font-medium flex items-center";
    if (reminder.message_type === 'appointment') {
      return <span className={`${baseClasses} bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200`}>
        <ClockIcon className="w-3 h-3 mr-1" />
        Appointment
      </span>
    } else {
      return <span className={`${baseClasses} bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200`}>
        <BellIcon className="w-3 h-3 mr-1" />
        Medication
      </span>
    }
  };
  
  // Render content based on current step
  const renderCurrentStep = () => {
    switch (currentStep) {
      case FlowStep.SEARCH:
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4 mt-4">
            <Card className="dark:border-gray-700">
              <CardBody>
                <div className="flex flex-col gap-4">
                  <h2 className="text-xl font-semibold">Reminder Details</h2>
                  <Select
                    label="Reminder Type"
                    labelPlacement="outside"
                    placeholder="Select reminder type"
                    value={type}
                    onChange={(e) => setType(e.target.value as 'appointment' | 'medication')}
                    startContent={
                      type === 'appointment' 
                        ? <ClockIcon className="w-4 h-4 text-blue-500" /> 
                        : <BellIcon className="w-4 h-4 text-purple-500" />
                    }
                  >
                    <SelectItem key="appointment">Appointment</SelectItem>
                    <SelectItem key="medication">Medication</SelectItem>
                  </Select>
                  <div className="grid grid-cols-2 gap-4">
                    <Input
                      label="Days Ahead"
                      labelPlacement="outside"
                      placeholder="Input days ahead"
                      type="number"
                      value={daysAhead.toString()}
                      onChange={(e) => setDaysAhead(parseInt(e.target.value) || 1)}
                      min={1}
                      startContent={<CalendarIcon className="w-4 h-4 text-gray-400" />}
                    />
                    <Input
                      label="Hours Ahead"
                      labelPlacement="outside"
                      placeholder="Input hours ahead"
                      type="number"
                      value={hoursAhead.toString()}
                      onChange={(e) => setHoursAhead(parseInt(e.target.value) || 1)}
                      min={1}
                      startContent={<ClockIcon className="w-4 h-4 text-gray-400" />}
                    />
                  </div>
                  <Button 
                    color="primary" 
                    onClick={handleSchedule}
                    isLoading={loading}
                  >
                    Find {type === 'appointment' ? 'Appointment' : 'Medication'} Reminders
                  </Button>
                  
                  {error && (
                    <div className="flex items-center text-red-500 dark:text-red-400 p-2 bg-red-50 dark:bg-red-900/30 rounded border border-red-200 dark:border-red-800">
                      <InformationCircleIcon className="w-5 h-5 mr-2" />
                      <span>{error}</span>
                    </div>
                  )}
                  
                  {searchPerformed && !error && reminders.length === 0 && (
                    <div className="flex items-center text-blue-500 dark:text-blue-400 p-2 bg-blue-50 dark:bg-blue-900/30 rounded border border-blue-200 dark:border-blue-800">
                      <InformationCircleIcon className="w-5 h-5 mr-2" />
                      <span>No reminders found. Try adjusting your search criteria.</span>
                    </div>
                  )}
                </div>
              </CardBody>
            </Card>
            
            {reminders.length > 0 && (
              <Card className="dark:border-gray-700">
                <CardBody>
                  <h2 className="text-xl font-semibold mb-4">Scheduled Reminders</h2>
                  <Table aria-label="Scheduled reminders" className="bg-white dark:bg-gray-800">
                    <TableHeader>
                      <TableColumn>Patient</TableColumn>
                      <TableColumn>Type</TableColumn>
                      <TableColumn width={200}>Details</TableColumn>
                      <TableColumn width={120}>Action</TableColumn>
                    </TableHeader>
                    <TableBody emptyContent="No reminders found">
                      {reminders.map((reminder, index) => (
                        <TableRow key={index}>
                          <TableCell>{reminder.patient_name}</TableCell>
                          <TableCell>{getStatusBadge(reminder)}</TableCell>
                          <TableCell>{formatReminderDetails(reminder)}</TableCell>
                          <TableCell>
                            <Button 
                              size="sm" 
                              color="primary" 
                              onClick={() => handleStartConversation(reminder)}
                            >
                              Manage
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardBody>
              </Card>
            )}
          </div>
        );
      
      case FlowStep.CONVERSATION:
        return (
          <div className="mt-4">
            {selectedReminder && (
              <div className="grid grid-cols-1 gap-4">
                <Card className="dark:border-gray-700">
                  <CardBody>
                    <div className="mb-4">
                      <h2 className="text-xl font-semibold flex items-center">
                        <span className="mr-2">
                          {selectedReminder.message_type === 'appointment' ? (
                            <ClockIcon className="h-5 w-5 text-blue-500" />
                          ) : (
                            <BellIcon className="h-5 w-5 text-purple-500" />
                          )}
                        </span>
                        {formatReminderType(selectedReminder.message_type)} Conversation
                      </h2>
                      <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
                        Patient: <span className="font-medium">{selectedReminder.patient_name}</span> | 
                        {selectedReminder.message_type === 'appointment' ? (
                          <span> Doctor: <span className="font-medium">{selectedReminder.details.doctor_name}</span></span>
                        ) : (
                          <span> Medication: <span className="font-medium">{selectedReminder.details.medication_name}</span></span>
                        )}
                      </p>
                    </div>
                    
                    {selectedReminder.message_type === 'appointment' ? (
                      <AppointmentConversation 
                        reminder={selectedReminder}
                        onComplete={handleCompleteConversation}
                      />
                    ) : (
                      <MedicationConversation 
                        reminder={selectedReminder}
                        onComplete={handleCompleteConversation}
                      />
                    )}
                  </CardBody>
                </Card>
              </div>
            )}
          </div>
        );
      
      case FlowStep.RESULTS:
        return (
          <div className="mt-4">
            {appointmentResult && selectedReminder && (
              <Card className="dark:border-gray-700">
                <CardBody>
                  <div className="flex items-center mb-4">
                    <CheckCircleIcon className="h-8 w-8 text-green-500 mr-2" />
                    <h2 className="text-xl font-semibold">
                      {selectedReminder.message_type === 'appointment' ? 'Appointment' : 'Medication'} Confirmed
                    </h2>
                  </div>
                  
                  {selectedReminder?.message_type === 'appointment' ? (
                    <div className="grid grid-cols-2 gap-4 mb-6">
                      <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-md">
                        <p className="text-sm text-gray-500 dark:text-gray-400">Patient</p>
                        <p className="font-medium">{selectedReminder.patient_name}</p>
                      </div>
                      <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-md">
                        <p className="text-sm text-gray-500 dark:text-gray-400">Doctor</p>
                        <p className="font-medium">{selectedReminder.details.doctor_name}</p>
                      </div>
                      <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-md">
                        <p className="text-sm text-gray-500 dark:text-gray-400">Date & Time</p>
                        <p className="font-medium">{new Date(selectedReminder.details.datetime).toLocaleString()}</p>
                      </div>
                      <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-md">
                        <p className="text-sm text-gray-500 dark:text-gray-400">Status</p>
                        <p className="font-medium text-green-600 dark:text-green-400">Confirmed</p>
                      </div>
                    </div>
                  ) : (
                    <div className="grid grid-cols-2 gap-4 mb-6">
                      <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-md">
                        <p className="text-sm text-gray-500 dark:text-gray-400">Patient</p>
                        <p className="font-medium">{selectedReminder.patient_name}</p>
                      </div>
                      <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-md">
                        <p className="text-sm text-gray-500 dark:text-gray-400">Medication</p>
                        <p className="font-medium">{selectedReminder.details.medication_name}</p>
                      </div>
                      <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-md">
                        <p className="text-sm text-gray-500 dark:text-gray-400">Dosage</p>
                        <p className="font-medium">{selectedReminder.details.dosage}</p>
                      </div>
                      <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-md">
                        <p className="text-sm text-gray-500 dark:text-gray-400">Status</p>
                        <p className="font-medium text-green-600 dark:text-green-400">
                          {appointmentResult.adherence || "Confirmed"}
                        </p>
                      </div>
                    </div>
                  )}
                  
                  {appointmentResult.audioAvailable && (
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-md mb-6 flex justify-between items-center">
                      <p className="text-blue-800 dark:text-blue-200 flex items-center">
                        <SpeakerWaveIcon className="w-5 h-5 mr-2" />
                        Audio recording available
                      </p>
                      <Button 
                        color="primary" 
                        size="sm" 
                        variant="flat"
                        startContent={<SpeakerWaveIcon className="w-4 h-4" />}
                        onClick={() => {
                          // Attempt to play the conversation audio
                          if (appointmentResult.audioData) {
                            import('../../app/api/audio').then(audio => {
                              audio.playAudioFromBase64(appointmentResult.audioData);
                            });
                          }
                        }}
                      >
                        Play Recording
                      </Button>
                    </div>
                  )}
                  
                  <div className="flex justify-between mt-4">
                    <Button 
                      color="default"
                      onClick={() => setCurrentStep(FlowStep.SEARCH)}
                      startContent={<ArrowPathIcon className="h-4 w-4" />}
                    >
                      Back to Schedule
                    </Button>
                    <Button 
                      color="primary"
                      onClick={() => {
                        setSelectedReminder(undefined);
                        setAppointmentResult(null);
                        setCurrentStep(FlowStep.SEARCH);
                      }}
                      startContent={<CheckCircleIcon className="h-4 w-4" />}
                    >
                      Done
                    </Button>
                  </div>
                </CardBody>
              </Card>
            )}
          </div>
        );
    }
  };
  
  return (
    <div className="flex flex-col gap-4">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Appointment Manager</h1>
        {currentStep !== FlowStep.SEARCH && (
          <Button 
            color="default" 
            variant="flat" 
            size="sm" 
            onClick={() => setCurrentStep(FlowStep.SEARCH)}
          >
            Back to Search
          </Button>
        )}
      </div>
      <Divider className="my-2" />
      
      {renderCurrentStep()}
    </div>
  );
}