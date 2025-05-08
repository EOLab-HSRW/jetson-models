import cv2
import base64
import numpy as np
import jetson_utils
import jetson_inference
from utils.utils import create_option
from models.base_model import BaseModel

class segnet(BaseModel):

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
    def filter_mode(self):
        return self.__filter_mode
    
    @property
    def is_custom(self):
        return self.__is_custom
#endregion

#region Methods

    def launch(self, data):
        try:

            self.__model_name = data.get('model_name')
            self.__variant = data.get('variant', 'fcn-resnet18-voc')
            self.__filter_mode = data.get('filter_mode', 'linear')
            self.__alpha = data.get('alpha', 150.0)
            self.__ignore_class = data.get('ignore_class', 'void')
            self.__visualize = data.get('visualize', 'overlay,mask')
            self.__is_custom = False
            self.__segnet = jetson_inference.segNet(self.__variant)
            self.__segnet.SetOverlayAlpha(self.__alpha)
            return True

        except Exception as e:
            print(f"Error initializing the model: {str(e)}")
            return False

    def run(self, img):
        img_height, img_width = img.shape[:2]

        cuda_img = jetson_utils.cudaFromNumpy(img)
        
        mask_overlay = jetson_utils.cudaAllocMapped(width=img_width, height=img_height, format='rgb8')
        
        self.__segnet.Process(cuda_img, ignore_class=self.__ignore_class)

        if 'overlay' in self.__visualize:
            self.__segnet.Overlay(mask_overlay, filter_mode=self.__filter_mode)
        
        segmentation_info = []
        base64_image_data = None

        if 'mask' in self.__visualize:
            mask_image = jetson_utils.cudaAllocMapped(width=img_width, height=img_height, format='gray8')
            self.__segnet.Mask(mask_image, filter_mode=self.__filter_mode)
            
            numpy_mask = jetson_utils.cudaToNumpy(mask_image)

            unique_classes, pixel_counts = np.unique(numpy_mask, return_counts=True)

            for class_id, count in zip(unique_classes, pixel_counts):
                if class_id != 0: 
                    segmentation_info.append({
                        "ClassID": int(class_id),
                        "ClassLabel": self.__segnet.GetClassDesc(int(class_id)),
                        "PixelCount": int(count)
                    })

        numpy_overlay = jetson_utils.cudaToNumpy(mask_overlay)

        _, encoded_img = cv2.imencode('.jpg', numpy_overlay)
        img_bytes = encoded_img.tobytes()

        base64_image_data = base64.b64encode(img_bytes).decode('utf-8')

        output_data = {
            "segmentation_info": segmentation_info,
            "image_data": base64_image_data
        }

        return output_data

    def stop(self):
        self.__segnet = None
        print("SegNet model stopped")

    @staticmethod
    def get_opts():
        info = {"segnet": {
            "description": "Segment a live camera stream using an image segmentation DNN.",
            "variant": create_option(
                typ=str,
                default="fcn-resnet18-voc",
                help="pre-trained model to load",
                options=["fcn-resnet18-voc", "fcn-resnet18-cityscapes", "fcn-resnet18-deepscene"]
            ),
            "filter_mode": create_option(
                typ=str,
                default="linear",
                help="filtering mode used during visualization",
                options=["point", "linear"]
            ),
            "alpha": create_option(
                typ=float,
                default=150.0,
                help="alpha blending value to use during overlay (0.0 to 255.0)"
            ),
            "ignore_class": create_option(
                typ=str,
                default="void",
                help="optional name of class to ignore in the visualization"
            ),
            "visualize": create_option(
                typ=str,
                default="overlay,mask",
                help="Visualization options (can be 'overlay' 'mask' 'overlay,mask')"
            )
        }}

        return info

#endregion
