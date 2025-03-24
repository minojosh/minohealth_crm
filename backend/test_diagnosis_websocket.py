"""
Test script for the diagnosis WebSocket service.
This script connects to the diagnosis WebSocket endpoint and simulates a conversation.
"""

import asyncio
import websockets
import json
import base64
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define WebSocket URL
WS_URL = os.getenv("NEXT_PUBLIC_WEBSOCKET_URL", "ws://localhost:8000")

async def test_diagnosis_websocket():
    """Test the diagnosis WebSocket with a simple conversation."""
    patient_id = 1  # Use an existing patient ID
    
    try:
        # Make sure we're not duplicating the 'ws' path
        base_url = WS_URL.rstrip('/')
        if '/ws' in base_url:
            uri = f"{base_url}/diagnosis/{patient_id}"
        else:
            uri = f"{base_url}/ws/diagnosis/{patient_id}"
            
        logger.info(f"Connecting to {uri}")
        
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to diagnosis WebSocket")
            
            # Receive the initial greeting
            response = await websocket.recv()
            response_data = json.loads(response)
            logger.info(f"Received greeting: {response_data}")
            
            # Send an initial context message
            context = "Patient presents with fever, cough, and shortness of breath for the past 3 days."
            await websocket.send(json.dumps({
                "type": "context",
                "context": context
            }))
            logger.info(f"Sent initial context: {context}")
            
            # Receive the response
            counter = 0
            while counter < 5:  # Limit the number of messages to process
                response = await websocket.recv()
                response_data = json.loads(response)
                logger.info(f"Received response: {response_data}")
                counter += 1
                
                # If we get a final message (not partial), break
                if response_data.get("type") == "message":
                    break
            
            # Send a follow-up question
            follow_up = "Does the patient have any history of asthma or respiratory issues?"
            await websocket.send(json.dumps({
                "type": "message",
                "text": follow_up
            }))
            logger.info(f"Sent follow-up question: {follow_up}")
            
            # Receive the response
            counter = 0
            while counter < 5:  # Limit the number of messages to process
                response = await websocket.recv()
                response_data = json.loads(response)
                logger.info(f"Received response: {response_data}")
                counter += 1
                
                # If we get a final message (not partial), break
                if response_data.get("type") == "message":
                    break
            
            # End the conversation
            await websocket.send(json.dumps({
                "type": "end_conversation"
            }))
            logger.info("Sent end conversation signal")
            
            # Receive the diagnosis summary
            counter = 0
            while counter < 10:  # Allow more messages for the summary
                response = await websocket.recv()
                response_data = json.loads(response)
                logger.info(f"Received summary response: {response_data}")
                counter += 1
                
                # If we get a final diagnosis summary, break
                if response_data.get("type") == "diagnosis_summary":
                    break
            
            logger.info("Test completed successfully")
            
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")

async def main():
    """Run the test."""
    await test_diagnosis_websocket()

if __name__ == "__main__":
    asyncio.run(main()) 