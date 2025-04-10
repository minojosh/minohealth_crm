import base64
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
import json
import time
from typing import Dict, List, Optional, Union
from pathlib import Path
from .XTTS_adapter import TTSClient
# Configure logging
logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversations with the OpenRouter API using OpenAI's client."""
    
    def __init__(self, api_key=None, base_url=None):
        """Initialize with OpenRouter API key and optional base URL."""
        config_path = Path(__file__).parent.parent / '.env'
        if not config_path.exists():
            logger.warning(f"Environment file not found at {config_path}, using environment variables")
        load_dotenv(dotenv_path=config_path, encoding='utf-8')
        
        # Get API key from parameters, environment, or raise error
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required")
        
        # Set base URL - default to OpenRouter's endpoint
        self.base_url = base_url or "https://openrouter.ai/api/v1"
        
        logger.info(f"Initializing ConversationManager with OpenRouter")
        
        # Initialize client
        try:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
            logger.info("Successfully initialized OpenAI client for OpenRouter")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
        
        self.tts = TTSClient(api_url=os.getenv("XTTS_SERVER_URL"))
        self.conversation_history = []
        self.custom_params = {"mode": "inference", "system_prompt": ''}
        self.model = "google/gemini-2.0-flash-lite-001"
        self.site_info = {
            "HTTP-Referer": os.getenv("SITE_URL", ""),
            "X-Title": os.getenv("SITE_NAME", "CRM")
        }

    def add_user_message(self, text=None, image_path=None):
        """Add a user message to the conversation history."""
        if not (text or image_path):
            raise ValueError("Either text or image must be provided")
        
        try:
            # Handle image+text message
            if image_path and text:
                with open(image_path, "rb") as image_file:
                    # Convert image to base64 for OpenRouter
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                
                self.conversation_history.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                        {"type": "text", "text": text},
                    ],
                })
                
            # Handle text-only message
            elif text:
                self.conversation_history.append({"role": "user", "content": text})
            
            return self
        except Exception as e:
            logger.error(f"Error in add_user_message: {e}")
            raise
    
    def get_assistant_response(self, max_tokens=300, temperature=1.0, stream=True, should_speak=False):
        """Get response from the assistant and update conversation history."""
        try:
            # Create request parameters
            request_params = {
                "model": self.model,
                "messages": self.conversation_history,
                "stream": stream,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "extra_headers": self.site_info
            }
            
            # Make the API call
            response = self.client.chat.completions.create(**request_params)
            
            # Handle streaming vs non-streaming
            if stream:
                return self._handle_streaming_response(response, should_speak)
            
            # Handle non-streaming
            content = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": content})
            
            # Only speak if enabled for non-streaming responses
            if should_speak:
                try:
                    self.tts.stream_text(content)
                    self.tts.wait_for_completion()
                    self.tts.stop()
                except Exception as e:
                    logger.error(f"Error in TTS for non-streaming response: {e}")
            
            return content
                
        except Exception as e:
            logger.error(f"Error in get_assistant_response: {e}")
            raise
    
    def _handle_streaming_response(self, response, should_speak=False):
        """Process streaming response and collect content."""
        collected_content = ""
        
        print("\nAssistant: ", end="", flush=True)
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                content_piece = chunk.choices[0].delta.content
                collected_content += content_piece
                
                print(content_piece, end="", flush=True)
                
                # Only stream to TTS if speaking is enabled
                if should_speak:
                    try:
                        self.tts.stream_text(content_piece)
                    except KeyboardInterrupt:
                        self.tts.stop()
                        pass
                        
        if should_speak:
            self.tts.wait_for_completion(2)      
            self.tts.stop()      
        print("\n")
        
        self.conversation_history.append({"role": "assistant", "content": collected_content})
        return collected_content
    
    def display_conversation(self):
        """Display conversation history for debugging."""
        for i, message in enumerate(self.conversation_history):
            role = message["role"]
            content = message["content"]
            
            # Handle different message formats
            if role == "user" and isinstance(content, list):
                text_content = next((item["text"] for item in content if item["type"] == "text"), "")
                print(f"[{i}] User: [Image] {text_content}")
            else:
                content_preview = content[:50] + "..." if isinstance(content, str) else "[Complex content]"
                print(f"[{i}] {role.capitalize()}: {content_preview}")

    def set_model(self, model_name):
        """Set the model to use for generation."""
        self.model = model_name
        return self

    def set_system_prompt(self, system_prompt):
        """Set the system prompt for the conversation."""
        if system_prompt:
            self.conversation_history = [{"role": "system", "content": system_prompt}] + [
                msg for msg in self.conversation_history if msg["role"] != "system"
            ]
        return self


def moremi(prompt, system_prompt="", max_tokens=300, temperature=1.0, model="google/gemini-2.0-flash-lite-001"):
    """Quick utility function to get a response from OpenRouter."""
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(dotenv_path):
        logger.warning(f"Environment file not found at {dotenv_path}, using environment variables")
    load_dotenv(dotenv_path=dotenv_path, encoding='utf-8')
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable required")
        
    try:
        conversation = ConversationManager(api_key=api_key)
        conversation.set_model(model)
        
        if system_prompt:
            conversation.set_system_prompt(system_prompt)
            
        return conversation.add_user_message(text=prompt).get_assistant_response(
            max_tokens=max_tokens, 
            temperature=temperature,
            stream=False
        )
    except Exception as e:
        logger.error(f"Error in moremi utility function: {e}")
        return f"Error: {str(e)}"


def generate_medical_report(image_path, context_data, country="Ghana", model="google/gemini-2.0-flash-lite-001"):
    """Generate a medical report from an image and context data."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable required")
        
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        # Read and convert image to base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        system_message = f"You are a medical AI assistant specialized in analyzing medical images from {country}. Generate a detailed medical report."
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", 
                 "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    {"type": "text", "text": context_data},
                 ]
                }
            ],
            stream=True,
            max_tokens=500,
            temperature=0.7,
            extra_headers={
                "HTTP-Referer": os.getenv("SITE_URL", "https://minohealth.ai"),
                "X-Title": os.getenv("SITE_NAME", "MinoHealth AI CRM")
            }
        )
        
        # Collect streamed content
        return "".join(
            chunk.choices[0].delta.content 
            for chunk in response 
            if chunk.choices[0].delta.content
        )
    except Exception as e:
        logger.error(f"Error generating medical report: {e}")
        raise