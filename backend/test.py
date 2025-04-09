from .moremi import ConversationManager
from .XTTS_adapter import TTSClient
from dotenv import load_dotenv
from pathlib import Path
import os

dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path, encoding='utf-8')
#tts_client = TTSClient(api_url=os.getenv("XTTS_URL") or os.getenv("SPEECH_SERVICE_URL"))
#tts_client = TTSClient("https://4185-34-34-51-11.ngrok-free.app/")
LLM = ConversationManager()

LLM.add_user_message("Tell me a story")

response = LLM.get_assistant_response()

# Use TTS for the reschedule message
# tts_client.TTS(
#     "Hello, I am your virtual assistant. How can I help you today?",
#     play_locally=True
# )
