from .moremi import ConversationManager

LLM = ConversationManager()

LLM.add_user_message("Hello, how are you?")

response = LLM.get_assistant_response(should_speak=True)
LLM.tts.wait_for_completion()
