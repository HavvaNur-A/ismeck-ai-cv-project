from .base import BaseModelWrapper
import torch
import timm
import numpy as np
from typing import List
from PIL import Image
import torchvision.transforms as T
from config import BACKBONE, DEVICE, logger



class ViTModelWrapper(BaseModelWrapper):
    def __init__(self, backbone: str=BACKBONE, num_classes: int=2, device: str=DEVICE):
        self.device = device
        self.backbone = backbone
        self.num_classes = num_classes
        self.model = timm.create_model(
            self.backbone, 
            pretrained=True, 
            num_classes=num_classes).to(self.device).eval()
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std =(0.229, 0.224, 0.225)
                    )
            ]
        )
        logger.info("Initialized ViTModelWrapper with backbone %s on %s.", self.backbone, self.device)
        
    def load_checkpoint(self, path: str):
        import torch
        ck = torch.load(path, map_location=self.device)
        if "model_state" in ck:
            self.model.load_state_dict(ck["model_state"])
        else:
            self.model.load_state_dict(ck)
        logger.info("Loaded checkpoint from %s.", path)
    
    def infer_batch(self, crops: List[np.ndarray]):
        if len(crops) == 0:
            return []
        xs = []
        for crop in crops:
            x = self.transform(crop)
            xs.append(x)
        batch = torch.stack(xs).to(self.device)
        with torch.no_grad():
            out = self.model(batch)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy().tolist()
        return probs
