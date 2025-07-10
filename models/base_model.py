import numpy as np
from typing import Dict, Any
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

    @property
    @abstractmethod
    def is_custom(self):
        pass

    @abstractmethod
    def launch(self, data: Dict[str, Any]):
        raise NotImplementedError()

    @abstractmethod
    def run(self, img: np.ndarray):
        raise NotImplementedError()
    
    @abstractmethod
    def stop(self):
        raise NotImplementedError()
    
    @staticmethod
    @abstractmethod
    def get_opts():
        raise NotImplementedError()