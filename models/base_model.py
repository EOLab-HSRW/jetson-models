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
    def launch(self, data):
        raise NotImplementedError()

    @abstractmethod
    def run(self, img):
        raise NotImplementedError()
    
    @abstractmethod
    def stop(self):
        raise NotImplementedError()
    
    @staticmethod
    @abstractmethod
    def get_opts():
        raise NotImplementedError()