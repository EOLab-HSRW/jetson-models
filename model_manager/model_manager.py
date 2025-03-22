import os
import cv2
import sys
import json
import gzip
import shutil
import random
import base64
import importlib
import websockets
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from pycocotools.coco import COCO
import xml.etree.ElementTree as ET
from models.base_model import BaseModel

# Dynamically added the submodule path
sys.path.append(os.path.abspath("vendor/pytorch_ssd"))
sys.path.append(os.path.abspath("vendor/pytorch_ssd/vision"))
sys.path.append(os.path.abspath("vendor/pytorch_ssd/vision/utils"))

import torch
from torch.utils.data import DataLoader
from vendor.pytorch_ssd.vision.datasets.voc_dataset import VOCDataset
from vendor.pytorch_ssd.vision.datasets.open_images import OpenImagesDataset
from vendor.pytorch_ssd.vision.ssd.vgg_ssd import create_vgg_ssd
from vendor.pytorch_ssd.vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vendor.pytorch_ssd.vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vendor.pytorch_ssd.vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vendor.pytorch_ssd.vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vendor.pytorch_ssd.vision.ssd.config import mobilenetv1_ssd_config
from vendor.pytorch_ssd.vision.ssd.config import vgg_ssd_config
from vendor.pytorch_ssd.vision.ssd.config import squeezenet_ssd_config
from vendor.pytorch_ssd.vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vendor.pytorch_ssd.vision.ssd.ssd import MatchPrior
from vendor.pytorch_ssd.vision.nn.multibox_loss import MultiboxLoss

train_transform = TrainAugmentation(300, [123,117,104], [1,1,1])
test_transform = TestTransform(300, [123,117,104], [1,1,1])

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

    async def receive_chunked_data(self, websocket, initial_metadata: dict) -> dict:
        """
        Helper function to read the remaining chunked data.
        Returns the final parsed JSON data, or None if there's an error.
        """

        received_data = b""
        expected_chunks = initial_metadata["total_chunks"]
        compressed = initial_metadata.get("compressed", False)
        chunks_received = 0

        print(f"Chunk metadata received: {initial_metadata}")

        while True:
            try:
                message = await websocket.recv()
            except websockets.ConnectionClosed:
                print("Connection closed while receiving chunks.")
                return None

            if isinstance(message, str):
                # We expect a “__END__” or an error here
                msg_json = json.loads(message)
                if msg_json.get("type") == "__END__":
                    print("Received __END__ message, finishing reception.")
                    break
                else:
                    print(f"Unexpected string message while reading chunks: {msg_json}")
                    await websocket.send(json.dumps({"error": "Unexpected string message in chunk mode"}))
                    return None
            else:
                # message is bytes → next chunk
                received_data += message
                chunks_received += 1
                print(f"Received chunk {chunks_received}/{expected_chunks}")

        if chunks_received != expected_chunks:
            print(f"Chunk count mismatch: expected {expected_chunks}, got {chunks_received}")
            await websocket.send(json.dumps(-1))
            return None

        # Decompress if needed
        if compressed:
            try:
                decompressed_data = gzip.decompress(received_data).decode()
                data = json.loads(decompressed_data)
                print(f"Successfully decompressed and loaded dataset with {len(data['dataset'])} images.")
            except Exception as e:
                print(f"Gzip decompression or JSON parsing failed: {str(e)}")
                await websocket.send(json.dumps(-1))
                return None
        else:
            try:
                data = json.loads(received_data.decode())
                print("Successfully loaded uncompressed dataset.")
            except Exception as e:
                print(f"JSON parsing failed: {str(e)}")
                await websocket.send(json.dumps(-1))
                return None

        if not "command" in data:
            await websocket.send(json.dumps(0))
            print("Error: Required ""command"" key does not exist into the data")
            return None

        # Acknowledge success
        await websocket.send(json.dumps(1))
        print(f"Dataset processing completed successfully from client {websocket.remote_address}.")
        return data

    def convert_coco_to_voc(self, coco, dataset_path, split_ratio=0.7):
        """
        Convert COCO JSON to Pascal VOC format.
        Splits dataset into 70% trainval and 30% test.
        """

        voc_path = os.path.join(dataset_path, "VOC")
        annotations_path = os.path.join(voc_path, "Annotations")
        images_path = os.path.join(voc_path, "JPEGImages")
        image_sets_path = os.path.join(voc_path, "ImageSets", "Main")

        # Create directories
        os.makedirs(annotations_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(image_sets_path, exist_ok=True)

        # Create trainval.txt and test.txt
        trainval_file = open(os.path.join(image_sets_path, "trainval.txt"), "w")
        test_file = open(os.path.join(image_sets_path, "test.txt"), "w")

        # Get all image IDs from COCO
        image_ids = list(coco.imgs.keys())

        # Shuffle and split 70% trainval / 30% test
        random.shuffle(image_ids)
        split_idx = int(len(image_ids) * split_ratio)
        train_ids = image_ids[:split_idx]
        test_ids = image_ids[split_idx:]

        for img_id in image_ids:
            img_info = coco.imgs[img_id]
            img_filename = img_info["file_name"]
            img_path = os.path.join(dataset_path, img_filename)

            # Copy image to VOC format directory
            shutil.copy(img_path, os.path.join(images_path, img_filename))

            # Create VOC XML annotation file
            xml_path = os.path.join(annotations_path, img_filename.replace(".jpg", ".xml"))
            root = ET.Element("annotation")

            ET.SubElement(root, "filename").text = img_filename

            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(img_info["width"])
            ET.SubElement(size, "height").text = str(img_info["height"])

            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = str(ann["category_id"])
                bbox = ann["bbox"]
                ET.SubElement(obj, "bndbox").text = f"{int(bbox[0])}, {int(bbox[1])}, {int(bbox[0]+bbox[2])}, {int(bbox[1]+bbox[3])}"

            tree = ET.ElementTree(root)
            tree.write(xml_path)

            # Write to trainval.txt (70%) or test.txt (30%)
            if img_id in train_ids:
                trainval_file.write(img_filename.replace(".jpg", "") + "\n")
            else:
                test_file.write(img_filename.replace(".jpg", "") + "\n")

        trainval_file.close()
        test_file.close()

    def convert_coco_to_openimages(self, coco, dataset_path, split_ratio=0.7):
        """
        Convert a COCO dataset into a simplified Open Images–style directory + CSV.
        """

        openimages_path = os.path.join(dataset_path, "OpenImages")
        os.makedirs(openimages_path, exist_ok=True)

        # Make subfolders for images
        train_images_dir = os.path.join(openimages_path, "train")
        test_images_dir = os.path.join(openimages_path, "test")
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(test_images_dir, exist_ok=True)

        category_mapping = {
            cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())
        }

        train_csv_file = os.path.join(openimages_path, "sub-train-annotations-bbox.csv")
        test_csv_file = os.path.join(openimages_path, "sub-test-annotations-bbox.csv")

        train_rows = []
        test_rows = []

        image_ids = list(coco.imgs.keys())
        random.shuffle(image_ids)
        split_idx = int(len(image_ids) * split_ratio)

        train_ids = set(image_ids[:split_idx])
        test_ids = set(image_ids[split_idx:])

        for img_id in image_ids:
            img_info = coco.imgs[img_id]
            filename = os.path.basename(img_info["file_name"])
            original_img_path = os.path.join(dataset_path, filename)
            if not os.path.exists(original_img_path):
                print(f"WARNING: Image {filename} not found. Skipping.")
                continue

            # Remove the file extension for ImageID in the CSV
            image_id_no_ext, _ = os.path.splitext(filename)

            if img_id in train_ids:
                dest_dir = train_images_dir
                subset_rows = train_rows
            else:
                dest_dir = test_images_dir
                subset_rows = test_rows

            # Copy image into train/ or test/
            dest_img_path = os.path.join(dest_dir, filename)
            shutil.copy2(original_img_path, dest_img_path)

            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            img_w = float(img_info["width"])
            img_h = float(img_info["height"])

            for ann in anns:
                x, y, w, h = ann["bbox"] 
                category_id = ann["category_id"]
                class_name = category_mapping.get(category_id, "unknown")

                # Normalize to 0..1
                x_min = x / img_w
                x_max = (x + w) / img_w
                y_min = y / img_h
                y_max = (y + h) / img_h

                row = [
                    image_id_no_ext,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    class_name
                ]
                subset_rows.append(row)

        cols = ["ImageID", "XMin", "XMax", "YMin", "YMax", "ClassName"]

        if train_rows:
            pd.DataFrame(train_rows, columns=cols).to_csv(train_csv_file, index=False)
        else:
            print("WARNING: No training images found.")

        if test_rows:
            pd.DataFrame(test_rows, columns=cols).to_csv(test_csv_file, index=False)
        else:
            print("WARNING: No testing images found.")

        print(f"OpenImages dataset generated: {len(train_rows)} train annotations, {len(test_rows)} test annotations.")

#region WebSocket Methods

    #WebSocket Inizialization
    async def start_server(self) -> None:
        """Start the WebSocket server"""

        print("WebSocket server is starting...")
        try:

            #Set the host ip
            host = "0.0.0.0"
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

                # Distinguish between chunk metadata or normal request
                if isinstance(message, str):
                    msg_json = json.loads(message)
                        
                    if 'total_chunks' in msg_json:
                        # ---- CHUNK MODE ----
                        data = await self.receive_chunked_data(
                            websocket,
                            initial_metadata=msg_json
                        )
                        # If there was an error, `receive_chunked_data` may return None:
                        if data is None:
                            print("Error: Recived data from the client is none")
                            return
                    else:
                        # ---- NORMAL MESSAGE MODE ----
                        data = msg_json

                elif isinstance(message, bytes):
                    print("error: First message not recognized as valid JSON")
                    await websocket.send(json.dumps(-1))
                    return
                else:
                    # ---- NORMAL MESSAGE MODE ----
                    data = json.loads(message)

                if not "command" in data:
                    print("Error: Required ""command"" key does not exist into the data")
                    await websocket.send(json.dumps(0))
                    return

                command = data["command"]
            
                # Print the message received from the client
                print(f"Message received from client {websocket.remote_address}: " "with the command: " f"{command}")

                if command == "/models":
                    await self.get_models(websocket)
                elif command == "/models/launch":
                    await self.launch(websocket, data)
                elif command == "/models/run":
                    await self.run(websocket, data)
                elif command == "/models/stop":
                    await self.stop(websocket, data)
                elif command == "/models/running":
                    await self.get_running(websocket)
                elif command == "/models/running/info":
                    await self.get_running_info(websocket)
                elif command == "/dataset":
                    await self.prepare_dataset(websocket, data)
                elif command == "/retrain":
                    await self.retrain_model(websocket, data)
                else:
                    print("error invalid action")
                    await websocket.send(json.dumps(-1))

        except websockets.ConnectionClosed as e:
            print(f"Client disconnected unexpectedly: {str(e)}")
            await websocket.send(json.dumps(-1))
        except Exception as e:
            print(f"Unhandled error: {str(e)}")
            await websocket.send(json.dumps(-1))

    #command to launch the model
    async def launch(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """
        Launch a model
        
        Required JSON keys:
        - "command": "/models/launch" indicate the action to launch a model
        - "model_name": Name of the model to launch (e.g., "detectnet")
        
        Optional keys (with defaults):
        - "variant_name": "ssd-mobilenet-v2"
        - "threshold": 0.5
        - "overlay": "box,labels,conf"

        If any required key is missing, returns -1. 
        On success, returns the launched model.
        """

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
                success = model_instance.launch(data)

                if success:
                    model_id = self.set_model_id()
                    self.running_models[model_id] = model_instance
                    response = model_id
                    print(f"Model: {model_name} launched successfully with the model_id: {model_id}")
                else:
                    response = -1
            
        except (ModuleNotFoundError, AttributeError):
            print(f"error: Model {model_name} is not supported or failed to load.")
            response = -1
            await websocket.send(json.dumps(response))
            return         

        except Exception as e:
            print(f"error: {str(e)}")
            response = -1
            await websocket.send(json.dumps(response))
            return

        await websocket.send(json.dumps(response))  

    #command to manage all the frames when a model is running
    async def run(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """
        Run a specific model with an image
        
        Required JSON keys:
        - "command": "/models/run" indicate the action to run a model
        - "image": base_64 string of the recieved image.
        - "model_id": Model ID of an launched model.

        If any required key is missing, returns -1 and for spelling error return 0. 
        On success, returns the result of the indicated model.
        """
    
        try:

            try:

                if len(self.running_models) == 0:
                    print("error no model is currently running")
                    response = 0
                else:
                    base64_img = data['image']
                    model_id = int(data['model_id'])

                    if model_id in self.running_models and base64_img not in (None, '', 'null'):
                        img_data = base64.b64decode(base64_img)
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

    #command to stop the current model
    async def stop(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """
        Stop a running model.
        
        Required JSON keys:
        - "command": "/models/stop" indicate the action to stop one or several models
        - "model_id": The model ID to stop, it can be an ID, a list of IDs or the string "all" or "ALL" to stops the models.  (e.g., 1000, [1000,1002,1025], "ALL")

        If any required key is missing, returns -1 and for spelling error, for no current running model, for an invalid model_id type return 0. 
        On success, returns ID or IDs of the stopped models.
        
        """
        
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

    #command to get a model_id list of the current running models
    async def get_running(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """
        Required JSON keys:
        - "command": "/models/running"

        Returns a model_id list of all the current running models.   

        If there are no running models returns 0.     
        """

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
            print(f"error: {str(e)}")
            response = -1
            await websocket.send(json.dumps(response))

    #command to know about the available models in the manager
    async def get_models(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """
        Required JSON keys:
        - "command": "/models"

        Returns a JSON with information about all available models inside the "/models" list folder.
        """

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
            print(f"error: {str(e)}")
            response = -1
            await websocket.send(json.dumps(response))

    #command to get all the current running models
    async def get_running_info(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """
        Required JSON keys:
        - "command": "/models/running/info"

        Returns a JSON with all the running models information
        """

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
            print(f"error: {str(e)}")
            response = -1
            await websocket.send(json.dumps(response))

    #command to prepare the dataset format
    async def prepare_dataset(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """
        Prepare the dataset for training using COCO format.
        
        Required JSON keys for all the images:
        - "command": "/dataset" indicate the action to create a dataset.
        - "model_name" : Indicate the model name for retrain (e.g., detectnet)
        - "id": The number ID of an image.
        - "image": The string base64 encoded image.
        - "class_label": The string class label of image (e.g., "person").
        - "dataset_name": The string name of the dataset.

        Optional keys (with defaults):
        - "BB": The scaled Bonding Boxes location. Its just required to retrain the detectnet model (e.g., {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5})

        JSON Format:
        {
            "command" : "/dataset",
            "model_name" : "detectnet, 
            "dataset": [
                {
                    "id": 1,
                    "image": "<base64_encoded_image_1>",
                    "BB": [
                        {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5},
                        {"x_min": 0.2, "y_min": 0.3, "x_max": 0.6, "y_max": 0.7}
                    ]
                },
                {
                    "id": 2,
                    "image": "<base64_encoded_image_2>",
                    "BB": [
                        {"x_min": 0.2, "y_min": 0.2, "x_max": 0.6, "y_max": 0.6}
                    ]
                }
            ],
            "class_label": "person",
            "dataset_name": "New_dataset"
        }

        If any required key is missing returns -1 or the current image will be ignored continue with the next one. 
        On success, returns the generated dataset name.
        """

        try:

            if not "model_name" in data:
                print("error: The requiered command or model_name key are missing")
                await websocket.send(json.dumps(0)) 
                return
            
            model_name = data["model_name"]

            response = 0
            model_name = model_name.lower()
            models_dir = Path(__file__).resolve().parent.parent / "models"

            for file in models_dir.glob("*py"):
                if model_name == str(file.stem).lower():

                    category_id = 1   #just a single category so far
                    dataset = list(data['dataset'])
                    class_label = data['class_label']
                    dataset_name = data['dataset_name']
                    annotation_id = 1

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
                            bounding_boxes = img['BB']

                            if not img_id or not base64_image:
                                continue

                            # Decode image and get image size (width, height)
                            img_data = base64.b64decode(base64_image)
                            np_arr = np.frombuffer(img_data, np.uint8)
                            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            img_height, img_width = img.shape[:2]

                            # Save the image in the dataset folder
                            img_path = tmp_dir / f"img_{img_id}.jpg"
                            with open(img_path, "wb") as img_file:
                                img_file.write(img_data)

                            # Add image information to COCO dataset
                            coco_data["images"].append({
                                "id": img_id,
                                "file_name": f"img_{img_id}.jpg",
                                "width": img_width,
                                "height": img_height
                            })

                            # Add bounding box annotation (only if DetectNet)
                            if bounding_boxes and model_name == "detectnet":

                                # Process multiple bounding boxes
                                for bbox in bounding_boxes:
                                    try:
                                        x_min = bbox["x_min"] * img_width
                                        y_min = bbox["y_min"] * img_height
                                        x_max = bbox["x_max"] * img_width
                                        y_max = bbox["y_max"] * img_height

                                        bbox_width = x_max - x_min
                                        bbox_height = y_max - y_min
                                        area = bbox_width * bbox_height

                                        if bbox_width <= 0 or bbox_height <= 0:
                                            print(f"Skipping invalid bounding box for image ID {img_id}")
                                            continue

                                        # Add bounding box annotation
                                        coco_data["annotations"].append({
                                            "id": annotation_id,
                                            "image_id": img_id,
                                            "category_id": category_id,
                                            "bbox": [
                                                x_min,  # Absolute x_min
                                                y_min,  # Absolute y_min
                                                bbox_width,
                                                bbox_height
                                            ],
                                            "area": area,
                                            "iscrowd": 0
                                        })
                                        annotation_id += 1  # Increment unique ID for each annotation
                                    except Exception as bbox_error:
                                        print(f"Error processing bounding box for image {img_id}: {bbox_error}")               

                        # Save COCO dataset to a JSON file
                        coco_path = tmp_dir / f"{dataset_name}.json"
                        with open(coco_path, "w") as coco_file:
                            json.dump(coco_data, coco_file, indent=4)

                        response = dataset_name

                    else:
                        response = -1
                        print(f"error: The dataset or class_label is missing")

            await websocket.send(json.dumps(response))

        except Exception as e:
            print(f"error: {str(e)}")
            response = -1
            await websocket.send(json.dumps(response)) 
    
    #command to retrain a model
    async def retrain_model(self, websocket, data: dict) -> None:
        """
        Retrain a model using a prepared dataset.  
        
        Required JSON keys:
        - "command": "/retrain" indicate the action to retrain an available model.
        - "model_name": Name of the model to retrain (e.g., "detectnet")
        - "dataset_name": Name of the dataset folder (e.g., "New_dataset")
        - "varitan_name": Name of the network architecture (e.g "mb2-ssd-lite")
        - "new_variant_name": New variant to use (e.g., "Fruits")
        
        Optional keys (with defaults):
        - "dataset_type": "open_images" (or "voc")
        - "epochs": 30
        - "batch_size": 4
        - "learning_rate": 0.01
        If any required key is missing, returns -1.
        On success, returns the variant_name.
        """

        # Validate required parameters
        for key in ["model_name", "dataset_name", "variant_name"]:
            if key not in data:
                await websocket.send(json.dumps(-1))
                return

        model_name = data["model_name"].lower()
        dataset_name = data["dataset_name"]
        variant_name = data["variant_name"]

        # Optional training hyperparameters
        dataset_type = data.get("dataset_type", "open_images")
        epochs = int(data.get("epochs", 30))
        batch_size = int(data.get("batch_size", 4))
        learning_rate = float(data.get("learning_rate", 0.01))

        # Path to the prepared dataset (created by prepare_dataset)
        dataset_path = os.path.join("datasets", dataset_name)
        coco_json_path = os.path.join(dataset_path, f"{dataset_name}.json")

        if not os.path.exists(dataset_path) or not os.path.exists(coco_json_path):
            print(f"Error: Dataset path {dataset_path} or JSON file {coco_json_path} does not exist.")
            await websocket.send(json.dumps(-1))
            return

        # Load COCO dataset
        coco = COCO(coco_json_path)

        # **Convert dataset based on user selection**
        if dataset_type == "voc":
            self.convert_coco_to_voc(coco, dataset_path)
            dataset_path = os.path.join(dataset_path, "VOC")
        elif dataset_type == "open_images":
            self.convert_coco_to_openimages(coco, dataset_path)
            dataset_path = os.path.join(dataset_path, "OpenImages")
        else:
            print(f"Error: Unsupported dataset type '{dataset_type}' provided.")
            await websocket.send(json.dumps(-1))
            return

        # **Load the converted dataset for training**
        if dataset_type == "voc":
            train_dataset = VOCDataset(dataset_path, transform=train_transform, target_transform=MatchPrior(mobilenetv1_ssd_config.priors,
                                            mobilenetv1_ssd_config.center_variance,
                                            mobilenetv1_ssd_config.size_variance, 0.5))
            val_dataset = VOCDataset(dataset_path, transform=test_transform, 
                                    target_transform=MatchPrior(mobilenetv1_ssd_config.priors,
                                            mobilenetv1_ssd_config.center_variance,
                                            mobilenetv1_ssd_config.size_variance, 0.5),
                                    is_test=True)
            config = vgg_ssd_config

        elif dataset_type == "open_images":
            train_dataset = OpenImagesDataset(dataset_path, transform=train_transform, 
                                                target_transform=MatchPrior(mobilenetv1_ssd_config.priors,
                                                mobilenetv1_ssd_config.center_variance,
                                                mobilenetv1_ssd_config.size_variance, 0.5),
                                                dataset_type="train")
            val_dataset = OpenImagesDataset(dataset_path, transform=test_transform, 
                                                target_transform=MatchPrior(mobilenetv1_ssd_config.priors,
                                                mobilenetv1_ssd_config.center_variance,
                                                mobilenetv1_ssd_config.size_variance, 0.5),
                                                dataset_type="test")
            config = mobilenetv1_ssd_config
    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # Instantiate the network based on model_name and variant_name.
        if model_name == "detectnet":
            if variant_name == "ssd-mobilenet-v1" or variant_name == "mb1-ssd":
                net = create_mobilenetv1_ssd
                config = mobilenetv1_ssd_config
            elif variant_name == "vgg16-ssd":
                net = create_vgg_ssd
                config = vgg_ssd_config
            elif variant_name == "mb1-ssd-lite":
                net = create_mobilenetv1_ssd_lite
                config = mobilenetv1_ssd_config
            elif variant_name == 'sq-ssd-lite':
                    net = create_squeezenet_ssd_lite
                    config = squeezenet_ssd_config
            elif variant_name == 'mb2-ssd-lite' or variant_name == "ssd-mobilenet-v2":
                net = create_mobilenetv2_ssd_lite
                config = mobilenetv1_ssd_config
            else:
                print(f"Variant {variant_name} not recognized for model {model_name}.")
                await websocket.send(json.dumps(-1))
                return
            config.set_image_size(300)
        else:
            print(f"Model {model_name} is not supported for retraining.")
            await websocket.send(json.dumps(-1))
            return

        # Set up device, loss function, and optimizer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                                center_variance=config.center_variance,
                                size_variance=config.size_variance, device=device)
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

        # Training loop (synchronous; consider running in a separate thread/task if needed)
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, net, criterion, optimizer, device, epoch)
            val_loss = self._validate_epoch(val_loader, net, criterion, device)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        save_directory = "/usr/local/bin/networks" #Default jetson network folder

        model_path = os.path.join(save_directory, f"{variant_name}.pth")
        net.save(model_path)
        print(f"Saved model {model_path}")

        # Return the variant_name as an indication of success.
        await websocket.send(json.dumps(variant_name))

    def _train_epoch(self, train_loader, net, criterion, optimizer, device, epoch):
        net.train()
        running_loss = 0.0
        num_batches = 0
        for i, data in enumerate(train_loader):
            # Unpack data (assumes dataset returns (image, boxes, labels))
            images, boxes, labels = data
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def _validate_epoch(self, val_loader, net, criterion, device):
        net.eval()
        running_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for data in val_loader:
                images, boxes, labels = data
                images = images.to(device)
                boxes = boxes.to(device)
                labels = labels.to(device)
                confidence, locations = net(images)
                regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
                loss = regression_loss + classification_loss
                running_loss += loss.item()
                num_batches += 1
        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

#endregion  
#endregion
