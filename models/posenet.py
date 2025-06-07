import jetson_utils
import jetson_inference
from utils.utils import create_option
from models.base_model import BaseModel


class posenet(BaseModel):
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
            self.__model_name = data["model_name"]
            self.__variant = data.get("variant_name", "resnet18-body")
            self.__overlay = data.get("overlay", "none") 
            self.__threshold = data.get("threshold", 0.15)
            self.__is_custom = False
            self.__net = jetson_inference.poseNet(self.variant, threshold=self.__threshold)
            return True
        except Exception as e:
            print(f"Error initializing the pose model: {str(e)}")
            return False

    def run(self, img):

        img_height, img_width = img.shape[:2]

        cuda_img = jetson_utils.cudaFromNumpy(img)
        poses = self.__net.Process(cuda_img, overlay=self.__overlay)
        
        pose_info = [{
            "Poses":{
                "Keypoints": [{
                    "ID": keypoint.ID,
                    "Name": self.__net.GetKeypointName(keypoint.ID),
                    "x": round(keypoint.x / img_width, 4),
                    "y": round(keypoint.y / img_height, 4)
                } for keypoint in pose.Keypoints],
                "Links": [{
                    "ID1": int(link[0]),
                    "ID2": int(link[1])
                }for link in pose.Links]
            }
            
        } for pose in poses] 

        return pose_info

    def stop(self):
        self.__net = None
        print("PoseNet model stopped")

    @staticmethod
    def get_opts():
        info = {
            "posenet": {
                "description": "Estimate human pose keypoints using PoseNet DNN.",
                "model": create_option(
                    typ=str,
                    default="resnet18-body",
                    help="pre-trained model to load",
                    options=["resnet18-body", "resnet18-hand", "resnet18-face"]
                ),
                "overlay": create_option(
                    typ=str,
                    default="none",
                    help="pose overlay flags (valid: 'links', 'keypoints', 'boxes', 'none')"
                ),
                "threshold": create_option(
                    typ=float,
                    default=0.15,
                    help="minimum detection threshold to use"
                )
            }
        }
        return info

#endregion