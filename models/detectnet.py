import sys
import jetson_utils
import jetson_inference
from models.model_base import ModelBase

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

        cuda_img = jetson_utils.cudaFromNumpy(img)

        detections = self.__detectnet.Detect(cuda_img)

        return detections   

    def stop(self):
        self.__is_running = False
        print("DetecNet model stopped")

