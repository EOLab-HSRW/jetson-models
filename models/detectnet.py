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
#endregion

#region Methods

    def launch(self, data):

        try:
            self.__model_name = data.get('model')
            self.__variant = data.get('variant', "ssd-mobilenet-v2")
            self.__threshold = data.get('threshold', 0.5)
            self.__overlay = data.get('overlay', 'box,labels,conf')

            self.__detectnet = jetson_inference.detectNet(network=self.__variant, threshold=self.__threshold)

        except Exception as e:
            print(f"Error inizializing the model: {str(e)}")

    def run(self, img):

        img_height, img_width = img.shape[:2]

        cuda_img = jetson_utils.cudaFromNumpy(img)

        detections = self.__detectnet.Detect(cuda_img, overlay=self.__overlay)

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

