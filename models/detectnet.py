import sys
import jetson_utils
import jetson_inference
from models.base_model import BaseModel

#DetectNet Class
class DetectNet(BaseModel):

    def __init__(self, network_name, threshold):
        super().__init__()
        self.__model_name = "detectnet"
        self.__network_name = network_name
        self.__detectnet = jetson_inference.detectNet(network_name, sys.argv, threshold)

    @property
    def model_name(self):
        return self.__model_name

    @property
    def network_name(self):
        return self.__network_name

    def run(self, img):

        img_height, img_width = img.shape[:2]

        cuda_img = jetson_utils.cudaFromNumpy(img)

        detections = self.__detectnet.Detect(cuda_img)

        detection_info = [{
                "detections":{
                    "ClassID": det.ClassID,
                    "Confidence": det.Confidence,
                    "BoundingBox": {
                        "Left": round(det.Left / img_width, 4),
                        "Top": round(det.Top / img_height, 4), 
                        "Right": round(det.Right / img_width, 4),
                        "Bottom": round(det.Bottom / img_height, 4)
                    }
                }
            } for det in detections]

        return detection_info   

    def stop(self):
        self.__detectnet = None
        print("DetecNet model stopped")

