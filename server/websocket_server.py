import json
import cv2
import base64
import numpy as np
from  model_manager.model_manager import ModelManager

# Create an instance of the model manager
manager = ModelManager()

# Available models list
MODELS = ['detectnet', 'imagenet']

#Handle actions with the client
async def handle_client(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        action = data.get('action')

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
        model_name = data.get('model')
        
        if model_name == "detectnet":
            threshold = data.get('threshold', 0.5)  # default = 0.5
            network_name = data.get('network', 'ssd-mobilenet-v2') # default ssd-mobilenet-v2
            response = manager.launch_model(model_name, network_name=network_name, threshold=threshold)
        elif model_name == "imagenet":
            network_name = data.get('network', 'googlenet') # default googlenet
            topK = data.get('topK', 1)  # default = 1
            response = manager.launch_model(model_name, network_name=network_name, topK=topK)
        else:
            raise ValueError(f"Model {model_name} is not supported")

        await websocket.send(json.dumps(response))
    
    except Exception as e:
        await websocket.send(json.dumps({"error": str(e)}))

#WebSocket to get the current model state
async def get_state(websocket):
    try:
        state = manager.get_state()
        await websocket.send(json.dumps({"state": state}))
    except Exception as e:
        await websocket.send(json.dumps({"error": str(e)}))

#WebSocket to know about the available models
async def get_info(websocket):
    try:
        await websocket.send(json.dumps({"models": MODELS}))
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
