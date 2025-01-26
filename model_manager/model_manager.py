import cv2
import json
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
            host = "192.168.1.25"
            port = 5000

            # Start the WebSocket server
            self.server = await websockets.serve(self.handle_client, host , port)
            print(f"Server running on ws://{host}:{port}")    
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
                elif path.startswith("/dataset/"):
                    model_name = path[len("/dataset/"):]
                    await self.prepare_dataset(websocket, model_name, data)
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
                    response = -1
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
                response = -1         
            await websocket.send(json.dumps(response))  

        except Exception as e:
            print(f"error: {e}")
            response = -1
            await websocket.send(json.dumps(response))

    #endpoint to manage all the frames when a model is running
    async def run(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """Run a specific model with an image"""
    
        try:

            try:

                if len(self.running_models) == 0:
                    print("error no model is currently running")
                    response = 0
                else:
                    img_str = data['image']
                    model_id = int(data['model_id'])

                    if model_id in self.running_models:
                        img_data = base64.b64decode(img_str)
                        np_arr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        response = self.running_models[model_id].run(img)
                    else:
                        print(f"error: No model is currently running with the model_id: {model_id}")
                        response = 0

            except Exception as e:
                print(f"error: Error during model execution: {str(e)}")
                response = -1

            await websocket.send(json.dumps(response))  
        except Exception as e:
            response = -1
            print(f"error {str(e)}")
            await websocket.send(json.dumps(response))

    #endpoint to stop the current model
    async def stop(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """Stop a running model."""
        
        try:

            if len(self.running_models) == 0: 
                print("No model is running")
                response = 0
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
                            response = 0
                            print(f"There is no model with the model_id: {model_id}")

                else:
                    print(f"Invalid model_id type: {type(model_id).__name__}. Expected str, int, or list.")
                    response = 0

            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"error: {e}")
            response = -1
            await websocket.send(json.dumps(response))

    #endpoint to get a model_id list of the current running models
    async def get_running(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """Return a model_id list of the current running models"""

        try:

            if  len(self.running_models) == 0:
                print("No model is running")
                response = 0
            else:

                models = []

                for id in sorted(self.running_models.keys()):
                    models.append(id)

                response = models

            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"error: {e}")
            response = -1
            await websocket.send(json.dumps(response))

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
            print(f"error: {e}")
            response = -1
            await websocket.send(json.dumps(response))

    #endpoint to get all the current running models
    async def get_running_info(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """Return all running models information"""

        try:

            if  len(self.running_models) == 0:
                print("model_id no model is running")
                response = 0
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
            print(f"error: {e}")
            response = -1
            await websocket.send(json.dumps(response))

    #endpoint to prepare the dataset format
    async def prepare_dataset(self, websocket: websockets.WebSocketServerProtocol, model_name: str, data: Dict[str, Any]) -> None:
        """Prepare the dataset for training using COCO format"""

        try:
            response = 0
            model_name = model_name.lower()
            models_dir = Path(__file__).resolve().parent.parent / "models"

            for file in models_dir.glob("*py"):
                if model_name == str(file.stem).lower():

                    category_id = 1   #just a single category so far
                    dataset = list(data['dataset'])
                    class_label = data['class_label']
                    dataset_name = data['dataset_name']

                    if dataset and class_label:
                        tmp_dir = Path(__file__).resolve().parent.parent / "datasets" / dataset_name
                        tmp_dir.mkdir(parents=True, exist_ok=True)  # Create a temporal folder in the project folder if not exists

                        # Initialize COCO dataset structure
                        coco_data = {
                            "images": [],
                            "annotations": [],
                            "categories": []
                        }

                        # Add category for class label, just a single class for dataset so far
                        coco_data["categories"].append({
                            "id": category_id,
                            "name": class_label
                        })

                        for img in dataset:
                            img_id = img['id']
                            base64_image = img['image']
                            bounding_box = img['BB']

                            if not img_id or not base64_image:
                                continue

                            # Decode image and get image size (width, height)
                            img_data = base64.b64decode(base64_image)
                            np_arr = np.frombuffer(img_data, np.uint8)
                            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            img_height, img_width = img.shape[:2]

                            # Save the image in the dataset folder
                            img_path = tmp_dir / f"{img_id}.jpg"
                            with open(img_path, "wb") as img_file:
                                img_file.write(img_data)

                            # Add image information to COCO dataset
                            coco_data["images"].append({
                                "id": img_id,
                                "file_name": f"{img_id}.jpg",
                                "width": img_width,
                                "height": img_height
                            })

                            # Add bounding box annotation (only if DetectNet)
                            if bounding_box and model_name == "detectnet":
                                x_min, y_min, x_max, y_max = (
                                    bounding_box["x_min"],
                                    bounding_box["y_min"],
                                    bounding_box["x_max"],
                                    bounding_box["y_max"],
                                )

                                if None in (x_min, y_min, x_max, y_max):
                                    print(f"Skipping invalid bounding box for image ID {img_id}")
                                    continue

                                bbox_width = x_max - x_min
                                bbox_height = y_max - y_min
                                area = bbox_width * bbox_height

                                coco_data["annotations"].append({
                                    "image_id": img_id,
                                    "category_id": category_id, 
                                    "bbox": [
                                        x_min / img_width,  
                                        y_min / img_height, 
                                        bbox_width / img_width,  
                                        bbox_height / img_height
                                    ]
                                })

                        # Save COCO dataset to a JSON file
                        coco_path = tmp_dir / f"{dataset_name}.json"
                        with open(coco_path, "w") as coco_file:
                            json.dump(coco_data, coco_file, indent=4)

                        response = dataset_name

            await websocket.send(json.dumps(response))

        except Exception as e:
            print(f"error: {e}")
            response = -1
            await websocket.send(json.dumps(response)) 

#endregion  

#endregion
