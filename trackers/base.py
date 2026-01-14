from abc import ABC, abstractmethod
from typing import List, Tuple

Track = Tuple[int, Tuple[int, int, int, int]]



class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections:List[Tuple[int,int,int,int,float]]) -> List[Track]:
        """Update tracker with detections and return active tracks"""
        raise NotImplementedError
