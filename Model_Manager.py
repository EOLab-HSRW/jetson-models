import base64
import cv2
import numpy as np
import sys
import asyncio
import websockets
import json
from abc import ABC, abstractmethod
import jetson_inference

#Template Abstract ModelBase
class ModelBase(ABC):

    @property
    @abstractmethod
    def is_running(self):
        pass

    @property
    @abstractmethod
    def model_name(self):
        pass

    @property
    @abstractmethod
    def network_name(self):
        pass

    @abstractmethod
    def run(self, img, overlay=None):
        raise NotImplementedError()
    
    def stop(self):
        raise NotImplementedError()

#DetectNet Class
class DetectNet(ModelBase):

    def __init__(self, network_name, threshold):
        super().__init__()
        self.__is_running = True
        self.__model_name = "detectnet"
        self.__network_name = network_name
        self.__detectnet = jetson_inference.detectNet(network_name, sys.argv, threshold)

    @property
    def is_running(self):
        return self.__is_running

    @property
    def model_name(self):
        return self.__model_name

    @property
    def network_name(self):
        return self.__network_name

    def run(self, img):

        detections = self.__detectnet.Detect(img)

        # Print the detection results
        print("Detected {:d} objects in image".format(len(detections)))
        for detection in detections:
            print(detection)

        return detections   

    def stop(self):
        self.__is_running = False
        print("DetecNet model stopped")

#ImageNet Class
class ImageNet(ModelBase):
    def __init__(self, network_name, topK=1):
        self.__is_running = False
        self.__model_name = "imagenet"
        self.__network_name = network_name
        self.__imagenet = jetson_inference.imageNet(network_name, sys.argv, topK)

    @property
    def is_running(self):
        return self.__is_running

    @property
    def model_name(self):
        return self.__model_name

    @property
    def network_name(self):
        return self.__network_name

    def run(self, img):

        predictions = self.__imagenet.Classify(img)

        # Print the detection results
        for classID, confidence in predictions:
            classLabel = self.__imagenet.GetClassLabel(classID)
            print(f"ImageNet: {confidence * 100:.2f}% class #{classID} ({classLabel})")

        return predictions    

    def stop(self):
        self.__is_running = False
        print("ImageNet model stopped")

class ModelManager:

    def __init__(self):
        self.running_model = None

    def launch_model(self, model_name, **kwargs):

        if self.running_model is not None:
            self.stop_model

        if model_name == "detectnet":
            self.running_model = DetectNet(kwargs["network_name"], kwargs.get("threshold", 0.5))

        elif model_name == "imagenet":
            self.running_model = ImageNet(kwargs["network_name"], kwargs.get("topK", 1))
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        print(f"{model_name} model launched")

    def run_model(self, img):
        self.running_model.run(self, img)

    def get_state(self):
        if not self.running_model:
            return {
                "is_running": "none",
                "model_name": None,
                "network_name": None
            }
        
        return {
            "is_running": self.running_model.is_running,
            "model_name": self.running_model.model_name,
            "network_name": self.running_model.network_name
        }
    
    def stop_model(self):
        if self.running_model is None: 
            return f'No model is running'

        self.running_model.stop()
        self.running_model = None
        return f'Model stopped successfully'
    
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
        network_name = data.get('network')
        threshold = data.get('threshold', 0.5)  # Only for detectNet
        topK = data.get('topK', 1)  # Only for imageNet
        
        if model_name == "detectnet":
            manager.launch_model(model_name, network_name=network_name, threshold=threshold)
        elif model_name == "imagenet":
            manager.launch_model(model_name, network_name=network_name, topK=topK)
        else:
            raise ValueError(f"Model {model_name} is not supported")

        await websocket.send(json.dumps({"message": f"{model_name} model launched successfully"}))
    
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
        await websocket.send(json.dumps({"message": str(response)}))
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
            detections = manager.running_model.run(img)

            detection_info = [{"ClassID": det.ClassID, "Confidence": det.Confidence, "BoundingBox": det.ROI}
                              for det in detections]
            
            await websocket.send(json.dumps({"detections": detection_info}))
        else:
            await websocket.send(json.dumps({"error": "No model is currently running"}))
    except Exception as e:
        await websocket.send(json.dumps({"error": str(e)}))

# Start the WebSocket server
start_server = websockets.serve(handle_client, "192.168.178.75", 5000)

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()