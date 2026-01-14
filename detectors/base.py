# farklı farklı algoritmaları da sistemin içerisine eklemek için aslında arka taraftaki senaryo da oluşturmuş olcaz
# sistemin içerisinde bu detector yapısının dönebileceği ifadelerin neler olduğunun bir kuralını buraya tanımlayıp bu kurala uygun olan bütün detector yapılarının içerisinde tanımlamak  
# ve sadece base'i çağırarak ve base'i bu yapıların hepsine initleyerek aslında bütün detector yapılarının base modülüyle çalıştığı ve base modülüne erişerek bütün detectors altındaki bütün yapılarak erişebildiğimiz bir senaryoyu inşa edecez.
# amaç, abstractmethod ile beraber

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

# each detection: (x1,y1,x2,y2,score) # yüzümüzün oluşturacağı bandigbox
Detection = Tuple[int, int, int, int, float]

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:# sistemde sürekli çalışacak bir frame vat
        """Detect objects in a frame and return list of detections."""
        raise NotImplementedError