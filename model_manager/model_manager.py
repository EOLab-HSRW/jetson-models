from models.detectnet import DetectNet
from models.imagenet import ImageNet

class ModelManager:

    def __init__(self):
        self.running_model = None

    def launch_model(self, model_name, **kwargs):

        if self.running_model is not None:
            if self.running_model.model_name == model_name:
                return {"message": f"The model {model_name} is already launched"}
            else:
                self.stop_model

        if model_name == "detectnet":
            self.running_model = DetectNet(kwargs["network_name"], kwargs["threshold"])

        elif model_name == "imagenet":
            self.running_model = ImageNet(kwargs["network_name"], kwargs["topK"])
        else:
            return {"message": f"{model_name} model not supported"}

        return {"message": f"{model_name} model launched successfully"}

    def run_model(self, img):
        return self.running_model.run(self, img)

    def get_state(self):
        if not self.running_model:
            return {
                "is_running": False,
                "model_name": None,
                "network_name": None
            }
        
        return {
            "is_running": True,
            "model_name": self.running_model.model_name,
            "network_name": self.running_model.network_name
        }
    
    def stop_model(self):
        if self.running_model is None: 
            return {"message": "No model is running"}

        model_name = self.running_model.model_name
        self.running_model.stop()
        self.running_model = None
        return {"message": f"The model {model_name} has been stopped successfully"}