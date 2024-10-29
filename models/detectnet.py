import jetson_utils
import jetson_inference
from models.base_model import BaseModel

#DetectNet Class
class detectnet(BaseModel):

#region Constructor

    def __init__(self, data):
            super().__init__()

            try:
                self.__model_name = data.get('model')
                self.__variant = data.get('variant', "ssd-mobilenet-v2")
                threshold = data.get('threshold', 0.5)
                self.__overlay = data.get('overlay', 'box,labels,conf')
                
                self.__detectnet = jetson_inference.detectNet(network=self.__variant, threshold=threshold)

            except Exception as e:
                print(f"Error inizializing the model: {str(e)}")

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

#endregion   

