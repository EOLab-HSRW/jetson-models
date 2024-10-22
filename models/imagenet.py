import sys
import jetson_utils
import jetson_inference
from models.model_base import ModelBase

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

        cuda_img = jetson_utils.cudaFromNumpy(img)

        predictions = self.__imagenet.Classify(cuda_img)

        return predictions

    def stop(self):
        self.__is_running = False
        print("ImageNet model stopped")
