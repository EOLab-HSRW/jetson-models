import jetson_utils
import jetson_inference
from models.base_model import BaseModel

#ImageNet Class
class imagenet(BaseModel):

#region Constructor

    def __init__(self, data):
        super().__init__()

        try:
            self.__model_name = data.get('model')
            self.__variant = data.get('variant', "googlenet")
            self.__topK = data.get('topK', 1)

            self.__imagenet = jetson_inference.imageNet(self.__variant)

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

        cuda_img = jetson_utils.cudaFromNumpy(img)

        predictions = self.__imagenet.Classify(cuda_img)

        classID, confidence = predictions

        classLabel = self.__imagenet.GetClassLabel(classID)

        predictions_info = [{
                "predictions": {
                    "ClassID": classID,
                    "ClassLabel": classLabel,
                    "Confidence": confidence * 100                 
                    }
                }]

        return predictions_info[:self.__topK]

    def stop(self):
        self.__imagenet = None
        print("imagenet model stopped")

#endregion
