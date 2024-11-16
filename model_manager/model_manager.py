import json
import cv2
import base64
import importlib
import websockets
import numpy as np
from pathlib import Path
from typing import Dict, Any
from models.base_model import BaseModel


class ModelManager:

#region Constructor
    def __init__(self) -> None:
        self.running_models: Dict[int, BaseModel] = {}
#endregion

#region Model Manager Methods

    def set_model_id(self) -> int:
        """Generate a unique model ID."""

        #First process id = 1000
        if len(self.running_models) == 0:
            return 1000
        
        current_id = 1000

        while current_id in self.running_models:
            current_id += 1
        return current_id

#region WebSocket Methods

    #WebSocket Inizialization
    async def start_server(self) -> None:
        """Start the WebSocket server"""

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
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str) -> None:
        """Handle incoming client messages"""

        try:            

            async for message in websocket:

                data = json.loads(message)

                # Print the message received from the client
                print(f"Message received from client {websocket.remote_address}: " "endpoint: " f"{path}")

                if path == "/models":
                    await self.get_models(websocket)
                elif path == "/models/launch":
                    await self.launch(websocket, data)
                elif path == "/models/run":
                    await self.run(websocket, data)
                elif path == "/models/stop":
                    await self.stop(websocket, data)
                elif path == "/models/running":
                    await self.get_running(websocket)
                elif path == "/models/running/info":
                    await self.get_running_info(websocket)
                else:
                    await websocket.send(json.dumps({"error": "Invalid action"}))

        except websockets.ConnectionClosed as e:
            print(f"Client {websocket.remote_address} disconnected: {e}")
        except json.JSONDecodeError:
            await websocket.send(json.dumps({"error": "Invalid JSON format"})) 
        except Exception as e:
            await websocket.send(json.dumps({"error": f"Unexpected error: {str(e)}"}))

    #endpoint to launch the model
    async def launch(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """Launch a model"""

        try:

            try:
                model_name = str(data['model']).lower()

                if model_name == "base_model": 
                    print("base_model is not supported")
                    response = 0
                else:
                    
                    #Getting the model class
                    model_module = importlib.import_module(f"models.{model_name}")
                    model_class = getattr(model_module, model_name)

                    # Instantiate a temporary object to call its info method
                    model_instance = model_class()
                    model_instance.launch(data)

                    model_id = self.set_model_id()
                    self.running_models[model_id] = model_instance
                    response = model_id
                    print(f"Model: {model_name} launched successfully with the model_id: {model_id}")
            
            except (ModuleNotFoundError, AttributeError):
                print(f"error: Model {model_name} is not supported or failed to load. Error: {e}")
                response = 0         
            await websocket.send(json.dumps(response))  

        except Exception as e:
            print(f"error: {e}")
            await websocket.send(str(0))

    #endpoint to manage all the frames when a model is running
    async def run(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """Run a specific model with an image"""
    
        try:

            try:

                if len(self.running_models) == 0:
                    response = {"error": "No model is currently running"}
                else:
                    img_str = data['image']
                    model_id = int(data['model_id'])

                    if model_id in self.running_models:
                        img_data = base64.b64decode(img_str)
                        np_arr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        response = self.running_models[model_id].run(img)
                    else:
                        response = {"error": f"No model is currently running with the model_id: {model_id}"}

            except Exception as e:
                response = {"error": f"Error during model execution: {str(e)}"}

            await websocket.send(json.dumps(response))  
        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

    #endpoint to stop the current model
    async def stop(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """Stop a running model."""
        
        try:

            if len(self.running_models) == 0: 
                print("No model is running")
                response = 1
            else:

                model_id = data['model_id']

                if isinstance(model_id, str) and model_id.lower() == "all":
                    stopped_models = []
                    for id in self.running_models:
                        stopped_models.append(id)
                        self.running_models[id].stop()
                    self.running_models.clear()
                    print("All the models have been stopped successfully")
                    response = stopped_models

                elif isinstance(model_id, int):
                    model_id = int(model_id)
                    if model_id in self.running_models:
                        model_name = self.running_models[model_id].model_name
                        response = model_id
                        self.running_models[model_id].stop()
                        del self.running_models[model_id]
                        print(f"Model: {model_name} with the model_id: {model_id} has been stopped successfully")
                    else:
                        print(f"There is no model with the model_id: {model_id}")
                        response = 0

                elif isinstance(model_id, list):

                    stopped_models = []

                    for id in model_id:
                        model_id = int(id)

                        if model_id in self.running_models:
                            model_name = self.running_models[model_id].model_name
                            stopped_models.append(model_id)
                            self.running_models[model_id].stop()
                            del self.running_models[model_id]
                            response = stopped_models

                            print(f"Model: {model_name} with the model_id {model_id} has been stopped successfully")
                        else:
                            print(f"There is no model with the model_id: {model_id}")

                else:
                    print(f"Invalid model_id type: {type(model_id).__name__}. Expected str, int, or list.")
                    response = 0

            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"error: {e}")
            await websocket.send(json.dumps(0))

    #endpoint to get a model_id list of the current running models
    async def get_running(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """Return a model_id list of the current running models"""

        try:

            if  len(self.running_models) == 0:
                print("No model is running")
                response = 1
            else:

                models = []

                for id in sorted(self.running_models.keys()):
                    models.append(id)

                response = models

            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"error: {e}")
            await websocket.send(json.dumps(0))

    #endpoint to know about the available models in the manager
    async def get_models(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """Return information about available models."""

        try:

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

            response = {"models": models_info}

            await websocket.send(json.dumps(response))
        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

    #endpoint to get all the current running models
    async def get_running_info(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """Return all running models information"""

        try:

            if  len(self.running_models) == 0:
                response = {"model_id": "No model is running"}
            else:

                models = {}

                for id in sorted(self.running_models.keys()):
                    models[id] = {
                        "model_name": self.running_models[id].model_name,
                        "variant": self.running_models[id].variant
                    }

                response = {"model_id": models}

            await websocket.send(json.dumps(response))
        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

#endregion  

#endregion
