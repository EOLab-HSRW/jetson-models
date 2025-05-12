import os
import cv2
import jetson_utils
import jetson_inference
from utils.utils import create_option
from models.base_model import BaseModel

#DetectNet Class
class detectnet(BaseModel):

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
            self.__variant = data.get('variant_name', "ssd-mobilenet-v2")
            self.__threshold = data.get('threshold', 0.5)
            self.__overlay = data.get('overlay', 'box,labels,conf')
            self.__is_custom = False
            
            # Built-in Jetson-inference model names
            predefined_models = [
                "ssd-mobilenet-v1", "ssd-mobilenet-v2", "ssd-inception-v2",
                "peoplenet", "peoplenet-pruned", "dashcamnet", "trafficcamnet", "facedetect"
            ]

            if self.__variant in predefined_models:
                print(f"[INFO] Launching built-in model: {self.__variant}")
                self.__detectnet = jetson_inference.detectNet(
                    network=self.__variant,
                    threshold=self.__threshold
                )

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
                self.__detectnet = jetson_inference.detectNet(
                    model=onnx_path,
                    labels=labels_path,
                    input_blob="input_0",
                    output_cvg="scores",
                    output_bbox="boxes",
                    threshold=self.__threshold
                )
                self.__overlay = 'none'
                self.__is_custom = True

            return True

        except Exception as e:
            print(f"[Error] Error inizializing the model: {str(e)}")
            return False

    def run(self, img):

        if self.is_custom:
            img = cv2.resize(img, (300,300))

        img_height, img_width = img.shape[:2]

        cuda_img = jetson_utils.cudaFromNumpy(img)

        detections = self.__detectnet.Detect(cuda_img, overlay=self.__overlay)

        detection_info = [{
                "detections":{
                    "ClassID": det.ClassID,
                    "ClassLabel": self.__detectnet.GetClassDesc(det.ClassID),
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
        print("detecnet model stopped")

    def get_opts():

        info = {"detectnet":{
                "description": "Locate objects in a live camera stream using an object detection DNN.",
                "variant": create_option(
                    typ = str,
                    default= "ssd-mobilenet-v2",
                    help="pre-trained model to load",
                    options=["ssd-mobilenet-v1", "ssd-mobilenet-v2", "ssd-inception-v2", "peoplenet", "peoplenet-pruned", "dashcamnet", "trafficcamnet", "facedetect"]
                ),
                "overlay": create_option(
                    typ = str,
                    default="box,labels,conf",
                    help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'",
                ),
                "threshold": create_option(
                    typ = float,
                    default= 0.5,
                    help="minimum detection threshold to use",
                )        
                }
            }

        return info

#endregion   

