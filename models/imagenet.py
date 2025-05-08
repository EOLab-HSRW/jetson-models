import jetson_utils
import jetson_inference
from utils.utils import create_option
from models.base_model import BaseModel

#ImageNet Class
class imagenet(BaseModel):

#region Constructor

    def __init__(self):
        super().__init__()

#endregion

#region Properties

    @property
    def model_name(self):
        return self.__model_name

    @property
    def variant(self):
        return self.__variant

    @property
    def is_custom(self):
        return self.__is_custom
#endregion

#region Methods

    def launch(self, data):

        try:
            self.__model_name = data.get('model_name')
            self.__variant = data.get('variant', "googlenet")
            self.__topK = data.get('topK', 1)
            self.__is_custom = False
            self.__imagenet = jetson_inference.imageNet(self.__variant)
            return True

        except Exception as e:
            print(f"Error inizializing the model: {str(e)}")
            return False

    def run(self, img):

        cuda_img = jetson_utils.cudaFromNumpy(img)

        predictions = self.__imagenet.Classify(cuda_img)

        classID, confidence = predictions

        classLabel = self.__imagenet.GetClassLabel(classID)

        predictions_info = [{
                "predictions": {
                    "ClassID": classID,
                    "ClassLabel": classLabel,
                    "Confidence": confidence                
                    }
                }]

        return predictions_info[:self.__topK]

    def stop(self):
        self.__imagenet = None
        print("imagenet model stopped")

    def get_opts():

        info = {"imagenet":{
                "description": "Classify a live camera stream using an image recognition DNN.",
                "variant": create_option(
                    typ = str,
                    default="googlenet",
                    help="Pre-trained model to load",
                    options=["alexnet", "googlenet", "googlenet-12", "resnet-18", "resnet-50", "resnet-101", "resnet-152", "vgg-16", "vgg-19", "inception-v4"]
                ),
                "topK": create_option(
                    typ = int,
                    default= 1,
                    help="show the topK number of class predictions"
                )
                }
            }

        return info

#endregion
