import jetson_inference
from utils.utils import create_option
from models.base_model import BaseModel
from utils.utils import get_str_from_dic
from utils.utils import get_float_from_dic
from utils.utils import get_cudaImgFromNumpy


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
            self.__model_name = get_str_from_dic(data, 'model_name', 'posernet')
            self.__variant = get_str_from_dic(data, 'variant_name', 'resnet18-body')
            self.__overlay = get_str_from_dic(data, 'overlay', 'none') 
            self.__threshold = get_float_from_dic(data, 'threshold', 0.15)
            self.__is_custom = False
            self.__net = jetson_inference.poseNet(self.variant, threshold=self.__threshold)
            return True
        except Exception as e:
            print(f"Error initializing the pose model: {str(e)}")
            return False

    def run(self, img):

        img_height, img_width = img.shape[:2]

        cuda_img = get_cudaImgFromNumpy(img)
        
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
        print(f"[INFO] PoseNet model with variant '{self.__variant}' has been stopped")
        self.__net = None

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