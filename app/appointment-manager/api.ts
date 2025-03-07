import { ReminderResponse, SchedulerRequest } from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const appointmentManagerApi = {
  /**
   * Start the scheduler process to find appointment or medication reminders
   */
  startScheduler: async (request: SchedulerRequest) => {
    try {
      const response = await fetch(`${API_BASE_URL}/start-scheduler`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json() as ReminderResponse[];
    } catch (error) {
      console.error('Error starting scheduler:', error);
      throw error;
    }
  },

  /**
   * Start a conversation for a specific reminder
   */
  startConversation: async (reminderId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/start-conversation/${reminderId}`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error starting conversation:', error);
      throw error;
    }
  },

  /**
   * Fetch all active reminders
   */
  getReminders: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/reminders`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json() as ReminderResponse[];
    } catch (error) {
      console.error('Error fetching reminders:', error);
      throw error;
    }
  },

  /**
   * Transcribe audio data through the WebSocket connection for a conversation
   * This is now handled directly by the WebSocket connection in our components
   */
  sendAudioForTranscription: async (base64Audio: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/transcribe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ audio: base64Audio }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error sending audio for transcription:', error);
      throw error;
    }
  },
};