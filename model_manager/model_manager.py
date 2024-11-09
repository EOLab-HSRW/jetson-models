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

    #endpoint to launch tthe model
    async def launch(self, websocket, data):
        try:

            try:
                model_name = str(data['model']).lower()

                if model_name == "base_model": 
                    response = {"message": f"Model {model_name} is not supported"}

                if self.running_model is not None and self.running_model.model_name == model_name:
                    response =  {"message": f"The model {model_name} is already launched"}

                #Getting the model class
                model_module = importlib.import_module(f"models.{model_name}")
                model_class = getattr(model_module, model_name)

                # Instantiate a temporary object to call its info method
                model_instance = model_class()
                model_instance.launch(data)

                self.running_model = model_instance

                response = {"message": f"{model_name} model launched successfully"}
            
            except (ModuleNotFoundError, AttributeError) as e:
                response = {"error": f"Model {model_name} is not supported or failed to load. Error: {str(e)}"}            

            await websocket.send(json.dumps(response))  

        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

    #endpoint to get the current model state
    async def get_state(self, websocket):

        try:

            if not self.running_model:
                response = {"state": {
                            "is_running": False,
                            "model_name": None,
                            "variant": None
                        } 
                    }
            else:
                response = {"state": {
                            "is_running": True,
                            "model_name": self.running_model.model_name,
                            "variant": self.running_model.variant
                        } 
                    }

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

    #endpoint to stop the current model
    async def stop(self, websocket):
        try:

            if self.running_model is None: 
                response = {"message": "No model is running"}
            else:
                model_name = self.running_model.model_name
                self.running_model.stop()
                self.running_model = None
                response = {"message": f"The model {model_name} has been stopped successfully"}

            await websocket.send(json.dumps(response))
        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

    #endpoint to manage all the frames when a model is running
    async def handle_image_frame(self, websocket, data):
        try:

            try:
                if not self.running_model:
                    response = {"error": "No model is currently running"}

                img_str = data['image']
                img_data = base64.b64decode(img_str)
                np_arr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                response = self.running_model.run(img)   
            except Exception as e:
                response = {"error": f"Error during model execution: {str(e)}"}

            await websocket.send(json.dumps(response))  
        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

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

                if action == "launch":
                    await self.launch(websocket, data)
                elif action == "get_state":
                    await self.get_state(websocket)
                elif action == "get_info":
                    await self.get_info(websocket)
                elif action == "stop":
                    await self.stop(websocket)
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

#endregion