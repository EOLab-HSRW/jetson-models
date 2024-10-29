from abc import ABC, abstractmethod

#Template Abstract BaseModel
class BaseModel(ABC):

    @property
    @abstractmethod
    def model_name(self):
        pass

    @property
    @abstractmethod
    def variant(self):
        pass

    @abstractmethod
    def run(self, img):
        raise NotImplementedError()
    
    def stop(self):
        raise NotImplementedError()
