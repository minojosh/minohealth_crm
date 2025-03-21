from .moremi import ConversationManager
from .XTTS_adapter import TTSClient
from dotenv import load_dotenv
from pathlib import Path
import os

dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path, encoding='utf-8')
tts_client = TTSClient(api_url=os.getenv("XTTS_URL") or os.getenv("SPEECH_SERVICE_URL"))

LLM = ConversationManager()

LLM.add_user_message("Tell me a short story")

response = LLM.get_assistant_response(should_speak=True)

# Use TTS for the reschedule message
# tts_client.TTS(
#     response,
#     play_locally=True
# )
