import sys
import jetson_utils
import jetson_inference
from models.base_model import BaseModel

#ImageNet Class
class ImageNet(BaseModel):
    def __init__(self, network_name, topK):
        self.__model_name = "imagenet"
        self.__network_name = network_name
        self.__imagenet = jetson_inference.imageNet(network_name, topK)

    @property
    def model_name(self):
        return self.__model_name

    @property
    def network_name(self):
        return self.__network_name

    def run(self, img):

        cuda_img = jetson_utils.cudaFromNumpy(img)

        predictions = self.__imagenet.Classify(cuda_img)

        classID, confidence = predictions
        classLabel = self.__imagenet.GetClassLabel(classID)

        predictions_info = [{
                "predictions": {
                    "ClassID": classID,
                    "ClassLabel": classLabel,
                    "Confidence": confidence * 100                 
                    }
                }]

        return predictions_info

    def stop(self):
        self.__imagenet = None
        print("ImageNet model stopped")
