import json
import cv2
import base64
import numpy as np
from  model_manager.model_manager import ModelManager

# Create an instance of the model manager
manager = ModelManager()

#Handle actions with the client
async def handle_client(websocket, path):
    async for message in websocket:

        data = json.loads(message)

        if isinstance(data, dict):
            action = data['action']
        else:
            action = data.get("action")

        # Print the message received from the client
        print(f"Message received from client {websocket.remote_address}: " "action: " f"{action}")

        if action == "launch_model":
            await launch_model(websocket, data)
        elif action == "get_state":
            await get_state(websocket)
        elif action == "get_info":
            await get_info(websocket)
        elif action == "stop_model":
            await stop_model(websocket)
        elif action == "image_frame":
            await handle_image_frame(websocket, data)
        else:
            await websocket.send(json.dumps({"error": "Invalid action"}))

#WebSocket to launch the model
async def launch_model(websocket, data):
    try:
        response = manager.launch_model(data)
        await websocket.send(json.dumps(response))  
    except Exception as e:
        await websocket.send(json.dumps({"error": str(e)}))

#WebSocket to get the current model state
async def get_state(websocket):
    try:
        response = manager.get_state()
        await websocket.send(json.dumps(response))
    except Exception as e:
        await websocket.send(json.dumps({"error": str(e)}))

#WebSocket to know about the available models
async def get_info(websocket):
    try:
        response = manager.models_info()
        await websocket.send(json.dumps(response))
    except Exception as e:
        await websocket.send(json.dumps({"error": str(e)}))

#WebSocket to stop the current model
async def stop_model(websocket):
    try:
        response = manager.stop_model()
        await websocket.send(json.dumps(response))
    except Exception as e:
        await websocket.send(json.dumps({"error": str(e)}))

#WebSocket to manage all the frames when a model is running
async def handle_image_frame(websocket, data):
    try:
        img_str = data['image']
        img_data = base64.b64decode(img_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if manager.running_model:
            result_info = manager.running_model.run(img)     
            await websocket.send(json.dumps(result_info))
        else:
            await websocket.send(json.dumps({"error": "No model is currently running"}))
    except Exception as e:
        await websocket.send(json.dumps({"error": str(e)}))
