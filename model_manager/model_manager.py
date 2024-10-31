import importlib
from pathlib import Path

class ModelManager:

    def __init__(self):
        self.running_model = None

    def launch_model(self, data):

        try:
            model_name = str(data.get('model')).lower()

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

    def run_model(self, img):
        if not self.running_model:
            return {"error": "No model is currently running"}
        
        try:
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

    def models_info(self):

        models_info = {}
        models_dir = Path(__file__).resolve().parent.parent / "models"

        for file in models_dir.glob("*.py"):

            model_name = str(file.stem).lower()
            if model_name == "base_model": 
                continue

            model_module = importlib.import_module(f"models.{model_name}")
            model_class = getattr(model_module, model_name) 
            model_info = model_class.info()

            # Add info of the model class to the dictionary
            models_info[model_name] = model_info[model_name]

        return {"models": models_info}