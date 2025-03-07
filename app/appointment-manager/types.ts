export interface SchedulerRequest {
  hours_ahead: number;
  days_ahead: number;
  type: 'appointment' | 'medication';
}

export interface ConversationResponse {
  status: string;
  message: string;
  conversation_id?: string;
}

export interface ReminderResponse {
  patient_name: string;
  message_type: string;
  details: Record<string, any>;
  timestamp: string;
}

export interface SpeechInputResponse {
  status: string;
  text: string;
}

export interface SpeechOutputResponse {
  status: string;
  message: string;
}