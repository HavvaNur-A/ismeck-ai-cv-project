from abc import ABC, abstractmethod
from typing import List
import numpy as np



class BaseModelWrapper(ABC):
    @abstractmethod
    def load_checkpoint(self, path: str): # sanki modelde dış model dosyası var da bunun checkpointi varmış da bunun üzerinden sistemi işletiyormuşum. 
        raise NotImplementedError
    
    @abstractmethod
    def infer_batch(self, crops:List[np.ndarray]) -> List[float]: #aamaç, pixellerle beraber bir frame içerisinde oluşturduğumuz görüntü üzerinden modelimiz işlediğinde bu frameler üzerinden bize bir score getirecek  
        """Return a list of scores/probabilities aligned with crops"""
        raise NotImplementedError
# modelimiz framei işlediğinde frame üzerinden score getirecek
# framede takip edilecek nesnesin sürekli o franeden ayrıştırması hakkında bir senaryo