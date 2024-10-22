from models.detectnet import DetectNet
from models.imagenet import ImageNet

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