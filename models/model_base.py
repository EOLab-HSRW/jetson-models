from abc import ABC, abstractmethod

#Template Abstract ModelBase
class ModelBase(ABC):

    @property
    @abstractmethod
    def is_running(self):
        pass

    @property
    @abstractmethod
    def model_name(self):
        pass

    @property
    @abstractmethod
    def network_name(self):
        pass

    @abstractmethod
    def run(self, img, overlay=None):
        raise NotImplementedError()
    
    def stop(self):
        raise NotImplementedError()
