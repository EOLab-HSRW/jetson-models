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
        self.running_models = {}
#endregion

#region Model Manager Methods

    def set_model_id(self) -> int:

        #First process id = 1000
        if len(self.running_models) == 0:
            return 1000
        
        current_id = 1000

        while current_id in self.running_models:
            current_id += 1
        return current_id

#region WebSocket Methods

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

                # Print the message received from the client
                print(f"Message received from client {websocket.remote_address}: " "endpoint: " f"{path}")

                if path == "/launch":
                    await self.launch(websocket, data)
                elif path == "/models":
                    await self.get_info(websocket)
                elif path == "/models/running":
                    await self.get_running(websocket)
                elif path == "/stop":
                    await self.stop(websocket, data)
                elif path == "/run":
                    await self.run(websocket, data)
                else:
                    await websocket.send(json.dumps({"error": "Invalid action"}))

        except websockets.ConnectionClosed as e:
            print(f"Client {websocket.remote_address} disconnected: {e}")
        except json.JSONDecodeError:
            await websocket.send(json.dumps({"error": "Invalid JSON format"})) 
        except Exception as e:
            await websocket.send(json.dumps({"error": f"Unexpected error: {str(e)}"}))

    #endpoint to launch the model
    async def launch(self, websocket, data):
        try:

            try:
                model_name = str(data['model']).lower()

                if model_name == "base_model": 
                    response = {"message": f"Model {model_name} is not supported"}

                #Getting the model class
                model_module = importlib.import_module(f"models.{model_name}")
                model_class = getattr(model_module, model_name)

                # Instantiate a temporary object to call its info method
                model_instance = model_class()
                model_instance.launch(data)

                self.running_models[self.set_model_id()] = model_instance

                response = {"message": f"{model_name} model launched successfully"}
            
            except (ModuleNotFoundError, AttributeError) as e:
                response = {"error": f"Model {model_name} is not supported or failed to load. Error: {str(e)}"}            

            await websocket.send(json.dumps(response))  

        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

    #endpoint to manage all the frames when a model is running
    async def run(self, websocket, data):
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
    async def stop(self, websocket, data):
        try:

            if len(self.running_models) == 0: 
                response = {"message": "No model is running"}
            else:

                model_id = data['model_id']

                if isinstance(model_id, str) and model_id.lower() == "all":
                    for id in self.running_models:
                        self.running_models[id].stop()
                    self.running_models.clear()
                    response = {"message": f"All the models have been stopped successfully"}

                elif isinstance(model_id, int):
                    model_id = int(model_id)
                    if model_id in self.running_models:
                        model_name = self.running_models[model_id].model_name
                        self.running_models[model_id].stop()
                        del self.running_models[model_id]
                        response = {"message": f"The model {model_name} has been stopped successfully"}
                    else:
                        response = {"message": f"There is no model with the model_id: {model_id}"}

                elif isinstance(model_id, list):

                    stopped_models = {}

                    for id in model_id:
                        p_id = int(id)

                        if p_id in self.running_models:
                            model_name = self.running_models[p_id].model_name
                            self.running_models[p_id].stop()
                            del self.running_models[p_id]
                            stopped_models[p_id] = f"The model {model_name} has been stopped successfully"
                        else:
                            stopped_models[p_id] = f"There is no model with the model_id: {p_id}"
                        
                    response = {"message": stopped_models}

                else:
                    response = {"message": f"Invalid model_id type: {type(model_id).__name__}. Expected str, int, or list."}

            await websocket.send(json.dumps(response))
        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

    #endpoint to get all the current running models
    async def get_running(self, websocket):

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

    #endpoint to know about the available models in the manager
    async def get_info(self, websocket):
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

#endregion  

#endregion
