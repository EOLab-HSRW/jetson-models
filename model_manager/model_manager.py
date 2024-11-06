import json
import cv2
import base64
import importlib
import websockets
import numpy as np
from pathlib import Path

class ModelManager:

#region Constructor
    def __init__(self):
        self.running_model = None
#endregion

#region Model Manager Methods

    def launch_model(self, data):

        try:
            model_name = str(data['model']).lower()

            if model_name == "base_model": 
                return {"message": f"Model {model_name} is not supported"}

            if self.running_model is not None and self.running_model.model_name == model_name:
                return {"message": f"The model {model_name} is already launched"}

            #Getting the model class
            model_module = importlib.import_module(f"models.{model_name}")
            model_class = getattr(model_module, model_name)

            # Instantiate a temporary object to call its info method
            model_instance = model_class()
            model_instance.launch(data)

            self.running_model = model_instance

            return {"message": f"{model_name} model launched successfully"}
        
        except (ModuleNotFoundError, AttributeError) as e:
            return {"error": f"Model {model_name} is not supported or failed to load. Error: {str(e)}"}

    def run_model(self, data):
        
        try:
            if not self.running_model:
                return {"error": "No model is currently running"}

            img_str = data['image']
            img_data = base64.b64decode(img_str)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            return self.running_model.run(img)   
        except Exception as e:
            return {"error": f"Error during model execution: {str(e)}"}

    def get_state(self):
        if not self.running_model:
            return {"state": {
                        "is_running": False,
                        "model_name": None,
                        "variant": None
                    } 
                }
        
        return {"state": {
                    "is_running": True,
                    "model_name": self.running_model.model_name,
                    "variant": self.running_model.variant
                } 
            }
    
    def stop_model(self):
        if self.running_model is None: 
            return {"message": "No model is running"}

        model_name = self.running_model.model_name
        self.running_model.stop()
        self.running_model = None
        return {"message": f"The model {model_name} has been stopped successfully"}

    def get_info(self):

        models_info = {}
        models_dir = Path(__file__).resolve().parent.parent / "models"

        for file in models_dir.glob("*.py"):

            model_name = str(file.stem).lower()
            if model_name == "base_model": 
                continue

            model_module = importlib.import_module(f"models.{model_name}")
            model_class = getattr(model_module, model_name) 
            model_info = model_class.get_opts()

            # Add info of the model class to the dictionary
            models_info[model_name] = model_info[model_name]

        return {"models": models_info}

#endregion  

#region WebSocket Server Methods

    #WebSocket Inizialization
    async def start_server(self):
        print("WebSocket server is starting...")
        try:

            #Set the host ip
            host = "192.168.178.75"

            # Start the WebSocket server
            self.server = await websockets.serve(self.handle_client, host , 5000)
            print(f"Server running on ws://{host}:5000")    
            await self.server.wait_closed()
            
        except Exception as e:
            print(f"Error starting the server: {e}")
        finally:
            print("Server shutdown.")

    #Handle actions with the client
    async def handle_client(self, websocket, path):

        try:
            async for message in websocket:
                data = json.loads(message)
                action = data['action']

                # Print the message received from the client
                print(f"Message received from client {websocket.remote_address}: " "action: " f"{action}")

                if action == "launch_model":
                    await self.ws_launch_model(websocket, data)
                elif action == "get_state":
                    await self.ws_get_state(websocket)
                elif action == "get_info":
                    await self.ws_get_info(websocket)
                elif action == "stop_model":
                    await self.ws_stop_model(websocket)
                elif action == "image_frame":
                    await self.handle_image_frame(websocket, data)
                else:
                    await websocket.send(json.dumps({"error": "Invalid action"}))

        except websockets.ConnectionClosed as e:
            print(f"Client {websocket.remote_address} disconnected: {e}")
        except json.JSONDecodeError:
            await websocket.send(json.dumps({"error": "Invalid JSON format"})) 
        except Exception as e:
            await websocket.send(json.dumps({"error": f"Unexpected error: {str(e)}"}))

    #WebSocket to launch the model
    async def ws_launch_model(self, websocket, data):
        try:
            response = self.launch_model(data)
            await websocket.send(json.dumps(response))  
        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

    #WebSocket to get the current model state
    async def ws_get_state(self, websocket):
        try:
            response = self.get_state()
            await websocket.send(json.dumps(response))
        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

    #WebSocket to know about the available models
    async def ws_get_info(self, websocket):
        try:
            response = self.get_info()
            await websocket.send(json.dumps(response))
        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

    #WebSocket to stop the current model
    async def ws_stop_model(self, websocket):
        try:
            response = self.stop_model()
            await websocket.send(json.dumps(response))
        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

    #WebSocket to manage all the frames when a model is running
    async def handle_image_frame(self, websocket, data):
        try:
            response = self.run_model(data)
            await websocket.send(json.dumps(response))  
        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

#endregion