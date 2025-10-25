import os
import io
import csv
import sys
import json
import gzip
import time
import shutil
import random
import base64
import asyncio
import argparse
import importlib
import websockets
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from pycocotools.coco import COCO
from utils.utils import delete_dir
import xml.etree.ElementTree as ET
from models.base_model import BaseModel
from utils.utils import BASE_NETWORKS_DIR

# Dynamically added the submodule path
sys.path.append(os.path.abspath("vendor/pytorch-ssd"))
sys.path.append(os.path.abspath("vendor/pytorch-ssd/vision"))
sys.path.append(os.path.abspath("vendor/pytorch-classification"))

from vision.ssd.ssd import MatchPrior
from vision.datasets.voc_dataset import VOCDataset
from vision.ssd.config import mobilenetv1_ssd_config
from vision.datasets.open_images import OpenImagesDataset
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

#from train import main

class ModelManager:

    #Global variables
    log_file = ""
    ws_user_ip = ""

#region Constructor
    def __init__(self) -> None:
        self.running_models: Dict[int, BaseModel] = {}
        self.log_queue = asyncio.Queue()
        self.model_lock = asyncio.Lock()
#endregion

#region Model Manager Methods

    def set_model_id(self) -> int:
        """Generate a unique model ID."""
        
        current_id = 1000

        while current_id in self.running_models:
            current_id += 1
        return current_id

    async def receive_chunked_data(self, websocket: websockets.WebSocketServerProtocol, initial_metadata: dict) -> dict:
        """
        Helper function to read the remaining chunked data.
        Returns the final parsed JSON data, or None if there's an error.
        """

        received_data = b""
        expected_chunks = initial_metadata["total_chunks"]
        compressed = initial_metadata.get("compressed", False)
        chunks_received = 0

        print(f"[INFO] Chunk metadata received: {initial_metadata}")

        while True:
            try:
                message = await websocket.recv()
            except websockets.ConnectionClosed:
                print("[ERROR] Connection closed while receiving chunks.")
                return None

            if isinstance(message, str):
                # We expect a “__END__” or an error here
                msg_json = json.loads(message)
                if msg_json.get("type") == "__END__":
                    print("[INFO] Received __END__ message, finishing reception.")
                    break
                else:
                    print(f"[ERROR] Unexpected string message while reading chunks: {msg_json}")
                    await websocket.send(json.dumps({"error": "Unexpected string message in chunk mode"}))
                    return None
            else:
                # message is bytes → next chunk
                received_data += message
                chunks_received += 1
                print(f"[INFO] Received chunk {chunks_received}/{expected_chunks}")

        if chunks_received != expected_chunks:
            print(f"[ERROR] Chunk count mismatch: expected {expected_chunks}, got {chunks_received}")
            await websocket.send(json.dumps(-1))
            return None

        # Decompress if needed
        if compressed:
            try:
                decompressed_data = gzip.decompress(received_data).decode()
                data = json.loads(decompressed_data)
                print(f"[INFO] Successfully decompressed and loaded dataset with {len(data['dataset'])} images.")
            except Exception as e:
                print(f"[ERROR] Gzip decompression or JSON parsing failed: {str(e)}")
                await websocket.send(json.dumps(-1))
                return None
        else:
            try:
                data = json.loads(received_data.decode())
                print("[INFO] Successfully loaded uncompressed dataset.")
            except Exception as e:
                print(f"[ERROR] JSON parsing failed: {str(e)}")
                await websocket.send(json.dumps(-1))
                return None

        if not "command" in data:
            await websocket.send(json.dumps(0))
            print("[ERROR] Required ""command"" key does not exist into the data")
            return None

        # Acknowledge success
        print(f"[INFO] Dataset processing completed successfully from client {websocket.remote_address}.")
        return data

    def convert_coco_to_voc(self, coco: COCO, dataset_path: str, split_ratio=0.7) -> None:
        """
        Convert COCO JSON to Pascal VOC format.
        Splits dataset into 70% trainval and 30% test.
        """

        voc_path = os.path.join(dataset_path, "VOC")
        dataset_images_dir_path = os.path.join(dataset_path, "images")
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

        # Create category ID → name mapping
        category_mapping = {
            cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())
        }

        all_classes = set()

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
            img_path = os.path.join(dataset_images_dir_path, img_filename)

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
                category_id = ann["category_id"]
                class_name = category_mapping.get(category_id)
                if class_name is None:
                    print(f"[INFO] Skipping unknown category_id {category_id} in image {img_filename}")
                    continue

                all_classes.add(class_name)

                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = class_name

                bbox = ann["bbox"]
                x_min = int(bbox[0])
                y_min = int(bbox[1])
                x_max = int(bbox[0] + bbox[2])
                y_max = int(bbox[1] + bbox[3])

                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(x_min)
                ET.SubElement(bndbox, "ymin").text = str(y_min)
                ET.SubElement(bndbox, "xmax").text = str(x_max)
                ET.SubElement(bndbox, "ymax").text = str(y_max)

            tree = ET.ElementTree(root)
            tree.write(xml_path)

            # Write to trainval.txt (70%) or test.txt (30%)
            if img_id in train_ids:
                trainval_file.write(img_filename.replace(".jpg", "") + "\n")
            else:
                test_file.write(img_filename.replace(".jpg", "") + "\n")

        trainval_file.close()
        test_file.close()

        # Save labels.txt
        labels_path = os.path.join(voc_path, "labels.txt")
        with open(labels_path, "w") as f:
            for class_name in sorted(all_classes):
                f.write(class_name + "\n")
        print(f"[INFO] Saved labels.txt with {len(all_classes)} classes.")

    def convert_coco_to_openimages(self, coco: COCO, dataset_path: str, split_ratio=0.7) -> None:
        """
        Convert a COCO dataset into a simplified Open Images–style directory + CSV.
        """

        dataset_images_dir_path = os.path.join(dataset_path, "images")
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
            original_img_path = os.path.join(dataset_images_dir_path, filename)
            if not os.path.exists(original_img_path):
                print(f"[WARN] Image {filename} not found. Skipping.")
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
                x_min = max(0.0, min(1.0, x / img_w))
                x_max = max(0.0, min(1.0, (x + w) / img_w))
                y_min = max(0.0, min(1.0, y / img_h))
                y_max = max(0.0, min(1.0, (y + h) / img_h))

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
            print("[WARN] No training images found.")

        if test_rows:
            pd.DataFrame(test_rows, columns=cols).to_csv(test_csv_file, index=False)
        else:
            print("[WARN] No testing images found.")

        print(f"[INFO] OpenImages dataset generated: {len(train_rows)} train annotations, {len(test_rows)} test annotations.")

    async def log_writer(self):
        global log_file
        fieldnames = [
            "Command", "Model_Name", "Variant_Name",
            "Start_Timestamp", "End_Timestamp", "Duration_Seconds",
            "Execution_Success", "User_IP"
        ]
        while True:
            log_data = await self.log_queue.get()
            try:
                write_header = not os.path.exists(log_file)
                with open(log_file, mode="a", newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(log_data)
            except Exception as e:
                print(f"[ERROR] Failed to write log: {e}")

#region WebSocket Methods

    #WebSocket Inizialization
    async def start_server(self, args: argparse.Namespace) -> None:
        """Start the WebSocket server"""

        if args.debug:
            print("[INFO] WebSocket server is starting in debug mode...")
            global log_file
            log_file = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            print("[INFO] WebSocket server is starting...")

        try:

            loop = asyncio.get_event_loop()
            loop.create_task(self.log_writer())

            #Set the host ip
            host = args.ip
            port = args.port

            # Start the WebSocket server
            self.server = await websockets.serve(
                lambda ws, path: self.handle_client(ws, path, args),
                host,
                port
            )
            print(f"[INFO] Server running on ws://{host}:{port}")    
            await self.server.wait_closed()
            
        except Exception as e:
            print(f"[ERROR] Error starting the server: {e}")
        finally:
            print("[INFO] Server shutdown.")

    #Handle actions with the client
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str, args: argparse.Namespace) -> None:
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
                            print("[INFO] Recived data from the client is none")
                            return
                    else:
                        # ---- NORMAL MESSAGE MODE ----
                        data = msg_json

                elif isinstance(message, bytes):
                    print("[ERROR] First message not recognized as valid JSON")
                    await websocket.send(json.dumps(-1))
                    return
                else:
                    # ---- NORMAL MESSAGE MODE ----
                    data = json.loads(message)

                if not "command" in data:
                    print("[WARN] Required ""command"" key does not exist into the data")
                    await websocket.send(json.dumps(0))
                    return

                command = data["command"]

                remote_ip, remote_port = websocket.remote_address
                global ws_user_ip
                ws_user_ip = remote_ip
            
                # Print the message received from the client
                print(f"[INFO] Message received from client {remote_ip}:{remote_port} : " "with the command: " f"{command}")

                if command == "/models":
                    await self.get_models(websocket, data, args)
                elif command == "/models/launch":
                    await self.launch(websocket, data, args)
                elif command == "/models/run":
                    await self.run(websocket, data, args)
                elif command == "/models/stop":
                    await self.stop(websocket, data, args)
                elif command == "/models/running":
                    await self.get_running(websocket, data, args)
                elif command == "/models/running/info":
                    await self.get_running_info(websocket, data, args)
                elif command == "/dataset":
                    await self.prepare_dataset_one_at_a_time(websocket, data, args)
                elif command == "/retrain":
                    await self.retrain_model(websocket, data, args)
                elif command == "/models/result":
                    await self.get_best_checkpoint(websocket, data, args)
                else:
                    print("[ERROR] Invalid action")
                    await websocket.send(json.dumps(-1))

        except websockets.ConnectionClosed as e:
            print(f"[ERROR] Client disconnected unexpectedly: {str(e)}")
            await websocket.send(json.dumps(-1))
        except Exception as e:
            print(f"[ERROR] Unhandled error: {str(e)}")
            await websocket.send(json.dumps(-1))

    #command to launch the model
    async def launch(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any], args: argparse.Namespace) -> None:
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
        
        start_time = time.time()
        start_dt = datetime.now()
        execution_success  = 0
        variant_name = "No variant"
        response = -1

        try:

            model_name = str(data['model_name']).lower()

            if model_name == "base_model": 
                print("[WARN] base_model is not supported")
                execution_success = 0
                response = 0
            else:
                    
                #Getting the model class
                model_module = importlib.import_module(f"models.{model_name}")
                model_class = getattr(model_module, model_name)

                # Instantiate a temporary object to call its info method
                model_instance = model_class()
                success = model_instance.launch(data)

                if success:
                    
                    async with self.model_lock:
                        model_id = self.set_model_id()
                        self.running_models[model_id] = model_instance
                    
                    variant_name = model_instance.variant
                    execution_success = 1
                    response = model_id
                    print(f"[INFO] Model: {model_name} launched successfully with the model_id: {model_id}")
                else:
                    response = -1
            
            await websocket.send(json.dumps(response))
            
        except (ModuleNotFoundError, AttributeError):
            print(f"[ERROR] Model {model_name} is not supported or failed to load.")
            response = -1
            await websocket.send(json.dumps(response))
            return         

        except Exception as e:
            print(f"[ERROR] {str(e)}")
            response = -1
            await websocket.send(json.dumps(response))
            return
        finally:
            if args.debug:
                global ws_user_ip
                end_time = time.time()
                end_dt = datetime.now()
                duration_seconds = round(end_time - start_time, 3)

                log_data = {
                    "Command": data["command"],
                    "Model_Name": model_name,
                    "Variant_Name": variant_name,
                    "Start_Timestamp": start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "End_Timestamp": end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "Duration_Seconds": duration_seconds,
                    "Execution_Success": execution_success,
                    "User_IP": ws_user_ip,
                }

                try:
                    await self.log_queue.put(log_data)
                except Exception as log_err:
                    print(f"[ERROR] Failed to enqueue log: {log_err}")  

    #command to manage all the frames when a model is running
    async def run(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any], args: argparse.Namespace) -> None:
        """
        Run a specific model with an image
        
        Required JSON keys:
        - "command": "/models/run" indicate the action to run a model
        - "image": base_64 string of the recieved image.
        - "model_id": Model ID of an launched model.

        If any required key is missing, returns -1 and for spelling error return 0. 
        On success, returns the result of the indicated model.
        """

        start_time = time.time()
        start_dt = datetime.now()
        execution_success  = 0
        response = -1

        try:

            if len(self.running_models) == 0:
                print("[WARN] No model is running...")
                execution_success = 0
                response = 0
            else:
                base64_img = data['image']
                model_id = int(data['model_id'])

                if model_id in self.running_models and base64_img not in (None, '', 'null'):
                    img_data = base64.b64decode(base64_img)
                    img_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
                    img = np.array(img_pil)
                    execution_success = 1

                    async with self.model_lock:
                        model = self.running_models.get(model_id)

                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(None, model.run, img)

                else:
                    print(f"[WARN] No model is currently running with the model_id: {model_id}")
                    execution_success = 0
                    response = 0

            await websocket.send(json.dumps(response))

        except Exception as e:
            execution_success = 0
            response = -1
            print(f"[ERROR] Error during model execution: {str(e)}")
            await websocket.send(json.dumps(response))
            return

        finally:
            if args.debug:
                global ws_user_ip
                end_time = time.time()
                end_dt = datetime.now()
                duration_seconds = round(end_time - start_time, 3)

                model_id = int(data.get("model_id", -1))
                if model_id in self.running_models:
                    model_name = self.running_models[model_id].model_name
                    variant_name = self.running_models[model_id].variant
                else:
                    model_name = "No model"
                    variant_name = "No variant"

                log_data = {
                    "Command": data["command"],
                    "Model_Name": model_name,
                    "Variant_Name": variant_name,
                    "Start_Timestamp": start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "End_Timestamp": end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "Duration_Seconds": duration_seconds,
                    "Execution_Success": execution_success,
                    "User_IP": ws_user_ip,
                }

                try:
                    await self.log_queue.put(log_data)
                except Exception as log_err:
                    print(f"[ERROR] Failed to enqueue log: {log_err}")

    #command to stop the current model
    async def stop(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any], args: argparse.Namespace) -> None:
        """
        Stop a running model.
        
        Required JSON keys:
        - "command": "/models/stop" indicate the action to stop one or several models
        - "model_id": The model ID to stop, it can be an ID, a list of IDs or the string "all" or "ALL" to stops the models.  (e.g., 1000, [1000,1002,1025], "ALL")

        If any required key is missing, returns -1 and for spelling error, for no current running model, for an invalid model_id type return 0. 
        On success, returns ID or IDs of the stopped models.
        
        """

        start_time = time.time()
        start_dt = datetime.now()
        execution_success  = 0
        response = -1
        stopped_models = []
        stopped_models_names = []
        stopped_variants = []
        
        try:

            if not self.running_models: 
                print("[WARN] No model is running...")
                execution_success = 1
                response = 0
            else:

                model_id = data['model_id']

                if isinstance(model_id, str) and model_id.lower() == "all":
                    async with self.model_lock:
                        for id in self.running_models:
                            stopped_models.append(id)
                            stopped_models_names.append(self.running_models[id].model_name)
                            stopped_variants.append(self.running_models[id].variant)
                            self.running_models[id].stop()
                        self.running_models.clear()
                        print("[INFO] All the models have been stopped successfully")
                        execution_success = 1
                        response = stopped_models

                elif isinstance(model_id, int):
                    model_id = int(model_id)
                    async with self.model_lock:
                        if model_id in self.running_models:
                            model_name = self.running_models[model_id].model_name
                            execution_success = 1
                            response = model_id
                            stopped_models_names.append(self.running_models[model_id].model_name)
                            stopped_variants.append(self.running_models[model_id].variant)
                            stopped_models.append(model_id)
                            self.running_models[model_id].stop()
                            del self.running_models[model_id]
                            print(f"[INFO] Model: {model_name} with the model_id: {model_id} has been stopped successfully")
                        else:
                            print(f"[WARN] There is no model with the model_id: {model_id}")
                            execution_success = 1
                            response = model_id

                elif isinstance(model_id, list):

                    async with self.model_lock:

                        for id in model_id:
                            model_id = int(id)

                            if model_id in self.running_models:
                                model_name = self.running_models[model_id].model_name
                                variant_name = self.running_models[model_id].variant
                                stopped_models.append(model_id)
                                stopped_models_names.append(model_name)
                                stopped_variants.append(variant_name)
                                self.running_models[model_id].stop()
                                del self.running_models[model_id]
                                print(f"[INFO] Model: {model_name} with variant: {variant_name} and with model_id: {model_id} has been stopped successfully")
                            else:
                                print(f"[INFO] There is no model with the model_id: {model_id}")

                    execution_success = 1 
                    response = stopped_models

                else:
                    print(f"[ERROR] Invalid model_id type: {type(model_id).__name__}. Expected str, int, or list.")
                    execution_success = 0
                    response = -1

            await websocket.send(json.dumps(response))

        except Exception as e:
            print(f"[ERROR] {e}")
            execution_success = 0
            response = -1
            await websocket.send(json.dumps(response))

        finally:
            if args.debug:
                global ws_user_ip
                end_time = time.time()
                end_dt = datetime.now()
                duration_seconds = round(end_time - start_time, 3)

                log_data = {
                    "Command": data["command"],
                    "Model_Name": "; ".join(map(str, stopped_models_names)),
                    "Variant_Name": "; ".join(map(str, stopped_variants)),
                    "Start_Timestamp": start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "End_Timestamp": end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "Duration_Seconds": duration_seconds,
                    "Execution_Success": execution_success,
                    "User_IP": ws_user_ip,
                }

                try:
                    await self.log_queue.put(log_data)
                except Exception as log_err:
                    print(f"[ERROR] Failed to enqueue log: {log_err}")
            
    #command to get a model_id list of the current running models
    async def get_running(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any], args: argparse.Namespace) -> None:
        """
        Required JSON keys:
        - "command": "/models/running"

        Returns a model_id list of all the current running models.   

        If there are no running models returns 0.     
        """

        start_time = time.time()
        start_dt = datetime.now()
        execution_success  = 0

        try:

            if not self.running_models:
                print("[WARN] No model is running...")
                execution_success  = 0
                response = 0
            else:

                models = []

                async with self.model_lock:
                    models = sorted(self.running_models.keys())

                execution_success  = 1
                response = models

            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            execution_success  = 0
            response = -1
            await websocket.send(json.dumps(response))
        finally:
           if args.debug:
                global ws_user_ip
                end_time = time.time()
                end_dt = datetime.now()
                duration_seconds = round(end_time - start_time, 3)

                log_data = {
                    "Command": data["command"],
                    "Model_Name": "",
                    "Variant_Name": "",
                    "Start_Timestamp": start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "End_Timestamp": end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "Duration_Seconds": duration_seconds,
                    "Execution_Success": execution_success,
                    "User_IP": ws_user_ip,
                }

                try:
                    await self.log_queue.put(log_data)
                except Exception as log_err:
                    print(f"[ERROR] Failed to enqueue log: {log_err}") 

    #command to know about the available models in the manager
    async def get_models(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any], args: argparse.Namespace) -> None:
        """
        Required JSON keys:
        - "command": "/models"

        Returns a JSON with information about all available models inside the "/models" list folder.
        """

        start_time = time.time()
        start_dt = datetime.now()
        execution_success  = 0
        response = -1

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

            execution_success  = 1
            response = {"models": models_info}

            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            execution_success  = 0
            response = -1
            await websocket.send(json.dumps(response))
        finally:
            if args.debug:
                global ws_user_ip
                end_time = time.time()
                end_dt = datetime.now()
                duration_seconds = round(end_time - start_time, 3)

                log_data = {
                    "Command": data["command"],
                    "Model_Name": "",
                    "Variant_Name": "",
                    "Start_Timestamp": start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "End_Timestamp": end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "Duration_Seconds": duration_seconds,
                    "Execution_Success": execution_success,
                    "User_IP": ws_user_ip,
                }

                try:
                    await self.log_queue.put(log_data)
                except Exception as log_err:
                    print(f"[ERROR] Failed to enqueue log: {log_err}")

    #command to get all the current running models
    async def get_running_info(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any], args: argparse.Namespace) -> None:
        """
        Required JSON keys:
        - "command": "/models/running/info"

        Returns a JSON with all the running models information
        """

        start_time = time.time()
        start_dt = datetime.now()
        execution_success  = 0
        response = -1

        try:

            if not self.running_models:
                print("[WARN] No model is running...")
                execution_success  = 0
                response = 0
            else:

                models = {}

                for id in sorted(self.running_models.keys()):
                    models[id] = {
                        "model_name": self.running_models[id].model_name,
                        "variant": self.running_models[id].variant
                    }

                execution_success  = 1
                response = {"model_id": models}

            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            execution_success  = 0
            response = -1
            await websocket.send(json.dumps(response))

        finally:
            if args.debug:
                global ws_user_ip
                end_time = time.time()
                end_dt = datetime.now()
                duration_seconds = round(end_time - start_time, 3)

                log_data = {
                    "Command": data["command"],
                    "Model_Name": "",
                    "Variant_Name": "",
                    "Start_Timestamp": start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "End_Timestamp": end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "Duration_Seconds": duration_seconds,
                    "Execution_Success": execution_success,
                    "User_IP": ws_user_ip,
                }

                try:
                    await self.log_queue.put(log_data)
                except Exception as log_err:
                    print(f"[ERROR] Failed to enqueue log: {log_err}") 

    #command to create a dataset with images at a time
    async def prepare_dataset_one_at_a_time(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any], args: argparse.Namespace) -> None:
        """
        Incrementally build a dataset for either DetectNet (object detection using COCO format) 
        or ImageNet (image classification using folder structure) by adding one image at a time.

        --------------------------
        DetectNet Dataset Format:
        --------------------------
        Creates a COCO-style JSON file + image folder structure.

        Required JSON keys:
        - "command": "/dataset"
        - "model_name": "detectnet"
        - "dataset_name": <string>
        - "class_label": <string>
        - "dataset": {
            "image": <base64-encoded JPEG>,
            "BB": [
                {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5},
                ...
            ]
        }

        DetectNet Dataset Layout:
        ```
        datasets/<dataset_name>/
        ├── images/
        └── <dataset_name>.json   ← COCO annotations
        ```

        --------------------------
        ImageNet Dataset Format:
        --------------------------
        Creates folder-of-folders format, where each class has its own subdirectory 
        inside 'train/', 'val/', or 'test/'

        Required JSON keys:
        - "command": "/dataset"
        - "model_name": "imagenet"
        - "dataset_name": <string>
        - "class_label": <string>  ← used as folder name (e.g., "dog", "cat")
        - "subset": "train" | "val" | "test"
        - "dataset": {
            "image": <base64-encoded JPEG>
        }

        ImageNet Dataset Layout:
        ```
        datasets/<dataset_name>/
        ├── train/
        │   ├── <class_label>/
        │   │   └── img_1.jpg
        ├── val/
        │   ├── <class_label>/
        │   │   └── img_1.jpg
        └── test/
        ```

        Returns via WebSocket:
        -  1 → success
        -  0 → missing fields or validation error
        - -1 → general error (e.g., decoding, saving, model not found)
        """

        start_time = time.time()
        start_dt = datetime.now()
        execution_success  = 0
        model_name = ""

        try:

            if "model_name" not in data:
                print("[WARN] Required model_name key is missing in the JSON payload.")
                execution_success  = 0
                await websocket.send(json.dumps(0)) 
                return            

            model_name = data["model_name"].lower()

            models_dir = Path(__file__).resolve().parent.parent / "models"
            model_exists = any(model_name == file.stem.lower() for file in models_dir.glob("*.py"))

            if not model_exists:
                print(f"[ERROR] Model '{model_name}' not found.")
                execution_success  = 0
                await websocket.send(json.dumps(-1)) 
                return

            if model_name == "detectnet":
                
                if "dataset_name" not in data or "class_label" not in data or "dataset" not in data:
                    print("[WARN] Required keys are missing in the JSON payload.")
                    execution_success  = 0
                    await websocket.send(json.dumps(0)) 
                    return   

                dataset_name = data["dataset_name"]
                class_label = data["class_label"]
                dataset_info = data["dataset"]
                image_data = dataset_info["image"]
                bboxes = dataset_info["BB"]

                dataset_path = Path("datasets") / dataset_name
                images_dir = dataset_path / "images"
                dataset_file = dataset_path / (dataset_name + ".json")

                images_dir.mkdir(parents=True, exist_ok=True)

                # Load or initialize annotations.json
                if dataset_file.exists():
                    with open(dataset_file, 'r') as f:
                        annotations = json.load(f)
                else:
                    annotations = {
                        "images": [],
                        "annotations": [],
                        "categories": []
                    }

                # Handle categories
                existing_category = next((cat for cat in annotations["categories"] if cat["name"] == class_label), None)
                if existing_category:
                    category_id = existing_category["id"]
                else:
                    category_id = max([cat["id"] for cat in annotations["categories"]], default=0) + 1
                    annotations["categories"].append({
                        "id": category_id,
                        "name": class_label
                    })

                # Generate new image ID
                image_id = max([img["id"] for img in annotations["images"]], default=0) + 1
                file_name = f"img_{image_id}.jpg"
                image_path = images_dir / file_name

                # Decode and save the image
                img = Image.open(BytesIO(base64.b64decode(image_data)))
                img.save(image_path)
                width, height = img.size

                annotations["images"].append({
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height
                })

                # Add bounding boxes
                next_ann_id = max([ann["id"] for ann in annotations["annotations"]], default=0) + 1
                for bbox in bboxes:
                    x_min = bbox["x_min"] * width
                    y_min = bbox["y_min"] * height
                    x_max = bbox["x_max"] * width
                    y_max = bbox["y_max"] * height
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    area = bbox_width * bbox_height

                    annotations["annotations"].append({
                        "id": next_ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": area,
                        "iscrowd": 0
                    })
                    next_ann_id += 1

                # Save updated annotations
                with open(dataset_file, 'w') as f:
                    json.dump(annotations, f, indent=4)

                print(f"status: success image_id: {image_id}")
                execution_success  = 1
                await websocket.send(json.dumps(1))

            elif model_name == "imagenet":

                try:
                    if "dataset_name" not in data or "class_label" not in data or "dataset" not in data:
                        print("[WARN] Required keys are missing in the JSON payload.")
                        execution_success  = 0
                        await websocket.send(json.dumps(0)) 
                        return 
                    
                    dataset_name = data["dataset_name"]
                    class_label  = data["class_label"]
                    subset       = data["subset"].lower()
                    if subset not in ("train", "val", "test"):
                        await websocket.send(json.dumps(0))
                        return

                    image_b64 = data["dataset"]["image"]
                    img_bytes = base64.b64decode(image_b64)

                    # Create the target directory if it doesn’t exist
                    base_dir    = Path("datasets") / dataset_name / subset / class_label
                    base_dir.mkdir(parents=True, exist_ok=True)               
                    
                    # Figure out the next image index in that folder
                    existing = list(base_dir.glob("img_*.jpg"))
                    if existing:
                        # get highest numeric suffix so far
                        idxs = [int(p.stem.split("_")[1]) for p in existing if "_" in p.stem]
                        next_id = max(idxs) + 1
                    else:
                        next_id = 1

                    filename = f"img_{next_id}.jpg"
                    save_path = base_dir / filename

                    with open(save_path, "wb") as f:
                        f.write(img_bytes)

                    print(f"status: success image_id: {filename}")
                    execution_success  = 1
                    await websocket.send(json.dumps(1))
                
                except Exception as e:
                    print(f"[ERROR] Dataset generation error")
                    await websocket.send(json.dumps(-1))
   
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            execution_success  = 0
            await websocket.send(json.dumps(-1))    
        finally:
            if args.debug:
                global ws_user_ip
                end_time = time.time()
                end_dt = datetime.now()
                duration_seconds = round(end_time - start_time, 3)

                log_data = {
                    "Command": data["command"],
                    "Model_Name": model_name,
                    "Variant_Name": "",
                    "Start_Timestamp": start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "End_Timestamp": end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "Duration_Seconds": duration_seconds,
                    "Execution_Success": execution_success,
                    "User_IP": ws_user_ip,
                }

                try:
                    await self.log_queue.put(log_data)
                except Exception as log_err:
                    print(f"[ERROR] Failed to enqueue log: {log_err}") 

    #command to train a model
    async def retrain_model(self, websocket: websockets.WebSocketServerProtocol, data: dict, args: argparse.Namespace) -> None:
        """
        Retrain a model using a prepared dataset.  
        
        Required JSON keys:
        - "command": "/retrain" indicate the action to retrain an available model.
        - "model_name": Name of the model to retrain (e.g., "detectnet")
        - "dataset_type": Specify dataset type. Currently supports voc and open_images.
        - "dataset_name": Name of the dataset folder (e.g., "Fruits_dataset")
        - "new_variant_name": New variant to use (e.g., "Fruits_SSD")
        
        Optional keys (with defaults):
        - "epochs": 30, Description: The number epochs
        - "batch_size": 1, Description: Batch size for training
        - "learning_rate": 0.01, Description: Initial learning rate 
        - "workers": 4, Description: Number of workers used in dataloading
        - "net": "mb1-ssd", Description: The network architecture, it can be mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.
        - "resolution": 300, Description: the NxN pixel resolution of the model (can be changed for mb1-ssd only)
        - "momentum": 0.9, Description: Momentum value for optim
        - "weight_decay": 5e-4, Description: Weight decay for SGD
        - "gama": 0.1, Description: Gamma update for SGD
        - "base_net_lr": 0.001, Description: Initial learning rate for base net, or None to use --lr
        - "extra_layers_lr": "None", Description: Initial learning rate for the layers not in base net and prediction heads
        - "scheduler": "cosine", Description: Scheduler for SGD. It can one of multi-step and cosine
        - "milestones": "80,100", Description: Milestones for MultiStepLR
        - "t_max": 100, Description: T_max value for Cosine Annealing Scheduler
        - "validation_epochs": 1, Description: The number epochs between running validation
        - "debug_steps": 10, Description: Set the debug log output frequency
        - "use_cuda": True, Description: Use CUDA to train model
        - "log_level": 'info', Description: Logging level, one of:  debug, info, warning, error, critical (default: info)
        - "pretrained_ssd": DEFAULT_PRETRAINED_MODEL, Description: Pre-trained base model

        If any required key is missing, returns -1.
        On success, returns the new_variant_name.
        """

        start_time = time.time()
        start_dt = datetime.now()
        execution_success  = 0
        model_name = ""
        new_variant_name = ""

        print("[INFO] Beginning retraining...")

        try:

            if "model_name" not in data:
                print("[WARN] Required model_name key is missing in the JSON payload.")
                execution_success  = 0
                await websocket.send(json.dumps(0)) 
                return            

            model_name = data["model_name"].lower()

            models_dir = Path(__file__).resolve().parent.parent / "models"
            model_exists = any(model_name == file.stem.lower() for file in models_dir.glob("*.py"))

            if not model_exists:
                print(f"[ERROR] Model '{model_name}' not found.")
                execution_success  = 0
                await websocket.send(json.dumps(-1)) 
                return

            if model_name == "detectnet":

                # Validate required parameters
                for key in ["dataset_type", "model_name", "dataset_name", "new_variant_name"]:
                    if key not in data:
                        print("[ERROR] Any required field is missing")
                        execution_success  = 0
                        await websocket.send(json.dumps(0))
                        return

                dataset_name = data["dataset_name"]
                new_variant_name = data["new_variant_name"]

                #Verify if there is already a model with the same name as the new variant
                if os.path.exists(os.path.join(BASE_NETWORKS_DIR, new_variant_name)):
                    print(f"[ERROR] A model already exists with the name {new_variant_name}")
                    execution_success  = 0
                    await websocket.send(json.dumps(0))
                    return

                base_new_model_directory = os.path.join(BASE_NETWORKS_DIR, new_variant_name)

                command_train = [
                    "python3", "vendor/pytorch-ssd/train_ssd.py",
                    "--model-dir", base_new_model_directory
                ]

                # Optional training hyperparameters
                if "dataset_type" in data:
                    dataset_type = data["dataset_type"]
                    command_train += ["--dataset-type", dataset_type]

                if "epochs" in data: command_train += ["--epochs", str(data["epochs"])]
                if "batch_size" in data: command_train += ["--batch-size", str(data["batch_size"])]
                if "learning_rate" in data: command_train += ["--learning-rate", str(data["learning_rate"])]            
                if "workers" in data: command_train += ["--workers", str(data["workers"])]
                if "net" in data: command_train += ["--net", data["net"]]
                if "resolution" in data: command_train += ["--resolution", str(data["resolution"])]
                if "momentum" in data: command_train += ["--momentum", str(data["momentum"])]
                if "weight_decay" in data: command_train += ["--weight-decay", str(data["weight_decay"])]
                if "gamma" in data: command_train += ["--gamma", str(data["gamma"])]
                if "base_net_lr" in data: command_train += ["--base-net-lr", str(data["base_net_lr"])]
                if "extra_layers_lr" in data: command_train += ["--extra-layers-lr", str(data["extra_layers_lr"])]
                if "scheduler" in data: command_train += ["--scheduler", str(data["scheduler"])]
                if "milestones" in data: command_train += ["--milestones", data["milestones"]]
                if "t_max" in data: command_train += ["--t-max", str(data["t_max"])]
                if "validation_epochs" in data: command_train += ["--validation-epochs", str(data["validation_epochs"])]
                if "debug_steps" in data: command_train += ["--debug-steps", str(data["debug_steps"])]
                if "use_cuda" in data: command_train += ["--use-cuda", str(data["use_cuda"])]
                if "log_level" in data: command_train += ["--log-level", data["log_level"]]
                if "pretrained_ssd" in data: command_train += ["--pretrained-ssd", data["pretrained_ssd"]]

                # Path to the prepared dataset (in the main folder so far)
                dataset_path = os.path.join("datasets", dataset_name)
                coco_json_path = os.path.join(dataset_path, f"{dataset_name}.json")

                if not os.path.exists(dataset_path) or not os.path.exists(coco_json_path):
                    print(f"[ERROR] Dataset path {dataset_path} or JSON file {coco_json_path} does not exist")
                    execution_success  = 0
                    delete_dir(os.path.join(BASE_NETWORKS_DIR, new_variant_name))
                    await websocket.send(json.dumps(-1))
                    return

                # Load COCO dataset
                coco = COCO(coco_json_path)

                # **Convert dataset based on user selection**
                if dataset_type == "voc":
                    print("[INFO] Converting dataset to a VOC format...")
                    self.convert_coco_to_voc(coco, dataset_path)
                    dataset_path = os.path.join(dataset_path, "VOC")
                elif dataset_type == "open_images":
                    print("[INFO] Converting dataset to a OpenImages format...")
                    self.convert_coco_to_openimages(coco, dataset_path)
                    dataset_path = os.path.join(dataset_path, "OpenImages")
                else:
                    print(f"[Error] Unsupported dataset type '{dataset_type}' provided.")
                    execution_success  = 0
                    delete_dir(os.path.join(BASE_NETWORKS_DIR, new_variant_name))
                    await websocket.send(json.dumps(-1))
                    return

                train_transform = TrainAugmentation(mobilenetv1_ssd_config.image_size, mobilenetv1_ssd_config.image_mean, mobilenetv1_ssd_config.image_std)
                target_transform = MatchPrior(mobilenetv1_ssd_config.priors, mobilenetv1_ssd_config.center_variance,  mobilenetv1_ssd_config.size_variance, 0.5)
                test_transform = TestTransform(mobilenetv1_ssd_config.image_size, mobilenetv1_ssd_config.image_mean, mobilenetv1_ssd_config.image_std)    

                # **Load the converted dataset for training**
                if dataset_type == "voc":
                    print("[INFO] Loading datasets for training and validation...")
                    train_dataset = VOCDataset(dataset_path, transform=train_transform, 
                                            target_transform=target_transform)
                    val_dataset = VOCDataset(dataset_path, transform=test_transform, 
                                            target_transform=target_transform, is_test=True)

                elif dataset_type == "open_images":
                    print("[INFO] Loading datasets for training and validation...")
                    train_dataset = OpenImagesDataset(dataset_path, transform=train_transform, 
                                                    target_transform=target_transform, dataset_type="train")
                    val_dataset = OpenImagesDataset(dataset_path, transform=test_transform, target_transform=target_transform, 
                                                    dataset_type="test")

                print(f"[INFO] Loaded Dataset with: {len(train_dataset)} train images and, {len(val_dataset)} for validation")

                num_classes = len(train_dataset.class_names)
                print(f"[INFO] Detected classes amount: {num_classes}")

                command_train += ["--datasets", dataset_path]

                try:
                    print("[INFO] Launching training subprocess...")
                    process = await asyncio.create_subprocess_exec(
                        *command_train,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT
                    )

                    async for line in process.stdout:
                        print("[TRAIN]", line.decode().strip())

                    await process.wait()

                    if process.returncode != 0:
                        print(f"[ERROR] Training failed with exit code {process.returncode}")
                        execution_success  = 0
                        delete_dir(os.path.join(BASE_NETWORKS_DIR, new_variant_name))
                        await websocket.send(json.dumps(-1))
                        return

                    print("[INFO] Training complete.")

                except Exception as e:
                    print(f"[ERROR] Unhandled training error: {str(e)}")
                    execution_success  = 0
                    delete_dir(os.path.join(BASE_NETWORKS_DIR, new_variant_name))
                    await websocket.send(json.dumps(-1))
                    return

                command_export = [
                    "python3", "vendor/pytorch-ssd/onnx_export.py",
                    "--model-dir", base_new_model_directory
                ]

                try:
                    print("[INFO] Exporting model subprocess...")
                    process = await asyncio.create_subprocess_exec(
                        *command_export,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT
                    )

                    async for line in process.stdout:
                        print("[EXPORT]", line.decode().strip())

                    await process.wait()

                    if process.returncode != 0:
                        print(f"[ERROR] Export failed with exit code {process.returncode}")
                        execution_success  = 0
                        delete_dir(os.path.join(BASE_NETWORKS_DIR, new_variant_name))
                        await websocket.send(json.dumps(-1))
                        return

                    print("[INFO] Exportation complete.")

                except Exception as e:
                    print(f"[ERROR] Unhandled export error: {str(e)}")
                    execution_success  = 0
                    delete_dir(os.path.join(BASE_NETWORKS_DIR, new_variant_name))
                    await websocket.send(json.dumps(-1))
                    return

                exported_model = os.path.join(BASE_NETWORKS_DIR, new_variant_name, "ssd-mobilenet.onnx")
                new_model_name = os.path.join(BASE_NETWORKS_DIR, new_variant_name, new_variant_name + ".onnx")

                exported_labels = os.path.join(BASE_NETWORKS_DIR, new_variant_name, "labels.txt")
                new_labels_name = os.path.join(BASE_NETWORKS_DIR, new_variant_name, new_variant_name + "_labels.txt")

                if os.path.exists(exported_model) and os.path.exists(exported_labels):
                    os.rename(exported_model, new_model_name)
                    os.rename(exported_labels, new_labels_name)
                    print(f"[INFO] Renamed model to {new_model_name}")
                    print(f"[INFO] Renamed labels to {new_labels_name}")

                execution_success  = 1 

                await websocket.send(json.dumps(new_variant_name))
                print("[INFO] Training is done!")
            
            elif model_name == "imagenet":
                
                for key in ["dataset_name", "new_variant_name"]:
                    if key not in data:
                        print("[ERROR] Missing required field(s) for ImageNet training.")
                        await websocket.send(json.dumps(0))
                        return
                    
                dataset_name = data["dataset_name"]
                new_variant_name = data["new_variant_name"]
                base_new_model_directory = os.path.join(BASE_NETWORKS_DIR, new_variant_name)
                dataset_path = os.path.join("datasets", dataset_name)
                arch = ""

                if not os.path.exists(dataset_path):
                    print(f"[ERROR] Dataset folder '{dataset_path}' not found.")
                    await websocket.send(json.dumps(-1))
                    return

                if not os.path.isdir(os.path.join(dataset_path, "train")) or not os.path.isdir(os.path.join(dataset_path, "val")):
                    print(f"[ERROR] Dataset must contain 'train/' and 'val/' folders.")
                    await websocket.send(json.dumps(-1))
                    return

                if os.path.exists(base_new_model_directory):
                    print(f"[ERROR] A model with name '{new_variant_name}' already exists.")
                    await websocket.send(json.dumps(0))
                    return

                os.makedirs(base_new_model_directory, exist_ok=True)

                command_train = [
                    "python3", "vendor/pytorch-classification/train.py",
                    dataset_path,
                    "--model-dir", base_new_model_directory
                ]

                # Optional training hyperparameters
                if "epochs" in data: command_train += ["--epochs", str(data["epochs"])]
                if "dataset_type" in data: command_train += ["--dataset-type", str(data["dataset_type"])]
                if "multi_label" in data: command_train += ["--multi-label", str(data["multi_label"])]
                if "multi_label_threshold" in data: command_train += ["--multi-label-threshold", str(data["multi_label_threshold"])]
                if "workers" in data: command_train += ["--workers", str(data["workers"])]
                if "start_epoch" in data: command_train += ["--start-epoch", str(data["start_epoch"])]
                if "batch_size" in data: command_train += ["--batch-size", str(data["batch_size"])]
                if "learning_rate" in data: command_train += ["--learning-rate", str(data["learning_rate"])]
                if "momentum" in data: command_train += ["--momentum", str(data["momentum"])]  
                if "weight_decay" in data: command_train += ["--weight-decay", str(data["weight_decay"])]
                if "resume" in data: command_train += ["--resume", str(data["resume"])] 
                if "pretrained" in data: command_train += ["--pretrained", str(data["pretrained"])]  
                if "resume" in data: command_train += ["--resume", str(data["resume"])] 
                if "seed" in data: command_train += ["--seed", str(data["seed"])]
                if "gpu" in data: command_train += ["--gpu", str(data["gpu"])]

                if "arch" in data: 
                    arch = str(data["arch"])
                    command_train += ["--arch", arch]
                else:
                    arch = "resnet18"

                try:
                    print("[INFO] Launching imagenet training subprocess...")
                    process = await asyncio.create_subprocess_exec(
                        *command_train,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT
                    )

                    async for line in process.stdout:
                        print("[TRAIN]", line.decode().strip())

                    await process.wait()

                    if process.returncode != 0:
                        print(f"[ERROR] Training failed with exit code {process.returncode}")
                        execution_success  = 0
                        delete_dir(base_new_model_directory)
                        await websocket.send(json.dumps(-1))
                        return

                    print("[INFO] Training complete.")

                    command_export = [
                        "python3", "vendor/pytorch-classification/onnx_export.py",
                        "--model-dir", base_new_model_directory
                    ]

                    try:
                        print("[INFO] Exporting model subprocess...")
                        process = await asyncio.create_subprocess_exec(
                            *command_export,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.STDOUT
                        )

                        async for line in process.stdout:
                            print("[EXPORT]", line.decode().strip())

                        await process.wait()

                        if process.returncode != 0:
                            print(f"[ERROR] Export failed with exit code {process.returncode}")
                            execution_success  = 0
                            delete_dir(base_new_model_directory)
                            await websocket.send(json.dumps(-1))
                            return

                        print("[INFO] Exportation complete.")

                    except Exception as e:
                        print(f"[ERROR] Unhandled export error: {str(e)}")
                        execution_success  = 0
                        delete_dir(base_new_model_directory)
                        await websocket.send(json.dumps(-1))
                        return

                    exported_model = os.path.join(BASE_NETWORKS_DIR, new_variant_name, arch + '.onnx')
                    new_model_name = os.path.join(BASE_NETWORKS_DIR, new_variant_name, new_variant_name + ".onnx")

                    exported_labels = os.path.join(BASE_NETWORKS_DIR, new_variant_name, "labels.txt")
                    new_labels_name = os.path.join(BASE_NETWORKS_DIR, new_variant_name, new_variant_name + "_labels.txt")

                    if os.path.exists(exported_model) and os.path.exists(exported_labels):
                        os.rename(exported_model, new_model_name)
                        os.rename(exported_labels, new_labels_name)
                        print(f"[INFO] Renamed model to {new_model_name}")
                        print(f"[INFO] Renamed labels to {new_labels_name}")

                    execution_success  = 1   

                    await websocket.send(json.dumps(new_variant_name))
                    print("[INFO] Training is done!")

                except Exception as e:
                    print(f"[ERROR] Unhandled training error: {str(e)}")
                    execution_success  = 0
                    delete_dir(base_new_model_directory)
                    await websocket.send(json.dumps(-1))
                    return

        except Exception as e:
            print(f"[ERROR] {str(e)}")
            execution_success  = 0
            await websocket.send(json.dumps(-1))
        finally:
            if args.debug:
                global ws_user_ip
                end_time = time.time()
                end_dt = datetime.now()
                duration_seconds = round(end_time - start_time, 3)

                log_data = {
                    "Command": data["command"],
                    "Model_Name": model_name,
                    "Variant_Name": new_variant_name,
                    "Start_Timestamp": start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "End_Timestamp": end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "Duration_Seconds": duration_seconds,
                    "Execution_Success": execution_success,
                    "User_IP": ws_user_ip,
                }

                try:
                    await self.log_queue.put(log_data)
                except Exception as log_err:
                    print(f"[ERROR] Failed to enqueue log: {log_err}")

    #command to get the lowest loss trained
    async def get_best_checkpoint(self, websocket: websockets.WebSocketServerProtocol, data: dict, args: argparse.Namespace) -> None:
        """
        Return the checkpoint file with the lowest loss for a given model variant name.

        Required JSON keys:
        - "variant_name": The name of the trained model variant.

        If found, returns the absolute path of the checkpoint with the lowest loss.
        If not found or an error occurs, returns -1.
        """

        start_time = time.time()
        start_dt = datetime.now()
        execution_success = 0
        variant_name = ""

        try:

            # Validate required parameter
            if "variant_name" not in data:
                print("[WARN] Any required field is missing")
                execution_success = 0
                await websocket.send(json.dumps(0))
                return
   
            variant_name = data["variant_name"]
            base_model_directory = os.path.join(BASE_NETWORKS_DIR, variant_name)

            best_loss = float('inf')
            best_checkpoint = None

            for file in os.listdir(base_model_directory):
                if not file.endswith(".pth"):
                    continue

                try:
                    # Extract the loss value from the filename
                    loss_str = file[file.rfind("-")+1 : -4]  # between last "-" and ".pth"
                    loss = float(loss_str)

                    if loss < best_loss:
                        best_loss = loss
                        best_checkpoint = os.path.join(base_model_directory, file)

                except ValueError:
                    # Skip files that don't follow the expected naming pattern
                    execution_success = 0
                    continue

            if best_checkpoint is None:
                execution_success = 0
                await websocket.send(json.dumps(-1))
                raise FileNotFoundError(f"No valid checkpoint with loss found in '{base_model_directory}'")

            print(f"[INFO] The path for the lowest loss checkpoint for model {variant_name} is: {best_checkpoint}")
            
            execution_success = 1
            await websocket.send(json.dumps(best_checkpoint))

        except Exception as e:
            print(f"[ERROR] {str(e)}")
            execution_success  = 0
            await websocket.send(json.dumps(-1))
        finally:
            if args.debug:
                global ws_user_ip
                end_time = time.time()
                end_dt = datetime.now()
                duration_seconds = round(end_time - start_time, 3)

                log_data = {
                    "Command": data["command"],
                    "Model_Name": "",
                    "Variant_Name": variant_name,
                    "Start_Timestamp": start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "End_Timestamp": end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "Duration_Seconds": duration_seconds,
                    "Execution_Success": execution_success,
                    "User_IP": ws_user_ip,
                }

                try:
                    await self.log_queue.put(log_data)
                except Exception as log_err:
                    print(f"[ERROR] Failed to enqueue log: {log_err}")

#endregion  
#endregion
