import numpy as np
from typing import Dict, Any
from abc import ABC, abstractmethod

#Template Abstract BaseModel
class BaseModel(ABC):

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def variant(self) -> str:
        pass

    @property
    @abstractmethod
    def is_custom(self) -> bool:
        pass

    @abstractmethod
    def launch(self, data: Dict[str, Any]) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def run(self, img: np.ndarray) -> any:
        raise NotImplementedError()
    
    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError()
    
    @staticmethod
    @abstractmethod
    def get_opts() -> dict:
        raise NotImplementedError()