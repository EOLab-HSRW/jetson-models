import os
import cv2
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
            self.__variant = data.get('variant_name', "googlenet")
            self.__topK = data.get('topK', 1)
            self.__is_custom = False

            # Built-in Jetson-inference model names
            predefined_models = ["alexnet", "googlenet", "googlenet-12",
                                 "resnet-18", "resnet-50", "resnet-101", 
                                 "resnet-152", "vgg-16", "vgg-19", "inception-v4"
            ]

            if self.__variant in predefined_models: 
                self.__imagenet = jetson_inference.imageNet(self.__variant)
            else:
                # Try to load custom ONNX model
                model_dir = os.path.join("/usr/local/bin/networks", self.__variant)
                onnx_path = os.path.join(model_dir, f"{self.__variant}.onnx")
                labels_path = os.path.join(model_dir, f"{self.__variant}_labels.txt")

                if not os.path.exists(onnx_path):
                    print(f"[ERROR] ONNX mdeol not found: {onnx_path}")
                    return False
                if not os.path.exists(labels_path):
                    print(f"[ERROR] Labels file not found: {labels_path}")
                    return False
                
                print(f"[INFO] Launching custom model from: {model_dir}")
                self.__imagenet = jetson_inference.imageNet(
                    model=onnx_path,
                    labels=labels_path,
                    input_blob="input_0",
                    output_blob="output_0"
                )
                self.__is_custom = True

            return True

        except Exception as e:
            print(f"Error inizializing the model: {str(e)}")
            return False

    def run(self, img):

        if self.is_custom:
            img = cv2.resize(img, (224,224))

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
        print(f"[INFO] ImageNet model with variant '{self.__variant}' has been stopped")
        self.__imagenet = None

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
