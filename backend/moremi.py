import base64
from openai import OpenAI
from typing import Optional


class ConversationManager:
    def __init__(self, base_url, api_key="None", system_prompt: Optional[str] = None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.conversation_history = []
        self.custom_params = {
            "mode": "inference"
        }
        if system_prompt is not None:
            self.custom_params["system_prompt"] = system_prompt
    
    def add_user_message(self, text=None, image_path=None):
        """Add a user message to the conversation history"""
        if image_path and text:
            # If both image and text are provided
            with open(image_path, "rb") as image_file: #use this only if you are sending a base 64
                image_data = image_file.read()
                encoded_image = base64.b64encode(image_data).decode('utf-8')
            
            self.conversation_history.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_path, #Note image_path should be aurl or a base 64
                            # set url to f"data:image/jpeg;base64,{encoded_image}" if you are sending a base64
                        },
                    },
                    {
                        "type": "text",
                        "text": text,
                    },
                ],
            })
        elif text:
            # Text-only message
            self.conversation_history.append({
                "role": "user",
                "content": text
            })
        else:
            raise ValueError("Text message is required")
        
        return self
    
    def get_assistant_response(self, max_tokens=300, temperature=1.0, stream=True):
        """Get response from the assistant and update conversation history"""
        response = self.client.chat.completions.create(
            model="workspace/merged-llava-model",
            messages=self.conversation_history,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
            #you can add other params supported by open ai
            extra_body=self.custom_params
        )
        
        if stream:
            # Handle streaming response
            collected_content = ""
            print("\nAssistant: ", end="", flush=True)
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content_piece = chunk.choices[0].delta.content
                    collected_content += content_piece
                    print(content_piece, end="", flush=True)
            
            print("\n")
            
            # Add assistant's response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": collected_content
            })
            
            return collected_content
        else:
            # Handle non-streaming response
            content = response.choices[0].message.content
            
            # Add assistant's response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": content
            })
            
            return content
    
    def display_conversation(self):
        """Display the entire conversation history"""
        for message in self.conversation_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user" and isinstance(content, list):
                # This is a message with an image
                text_content = next((item["text"] for item in content if item["type"] == "text"), "")
                print(f"User: [Image] {text_content}")
            else:
                print(f"{role.capitalize()}: {content}")


# Example usage
def main():
    # Initialize conversation manager
    conversation = ConversationManager(base_url="http://46.101.91.139:5003/v1")
    
    text="Tell me my name"
    
    #text only follow-up
    conversation.add_user_message(
        text = text
        
    )
    
    # Get second response
    conversation.get_assistant_response()
    
    
    # Display full conversation
    print("\nFull Conversation:")
    conversation.display_conversation()


if __name__ == "__main__":
    main()



