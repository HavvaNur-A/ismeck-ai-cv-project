# detectors/mtcnn_detector.py
from .base import BaseDetector, Detection
import numpy as np
from config import logger, DEVICE
try:
    # facenet-pytorch
    from facenet_pytorch import MTCNN
    _HAS_MTCNN = True
except Exception as e:
    logger.warning("facenet-pytorch import failed: %s. MTCNN detector unavailable.", e)
    _HAS_MTCNN = False
    
class MTCNNDetector(BaseDetector):
    """
    Wrapper around facenet_pytorch.MTCNN to provide detections as
    [(x1,y1,x2,y2,score), ...] in integer pixel coordinates.
    """
    def __init__(self, device: str = DEVICE, keep_all: bool = True, post_process: bool = True, min_face_size: int = 40):
        if not _HAS_MTCNN:
            logger.warning("MTCNNDetector initialized but facenet-pytorch not available.")
            self.mtcnn = None
        else:
            # device should be 'cuda' or 'cpu'
            self.mtcnn = MTCNN(keep_all=keep_all, device=device, post_process=post_process, min_face_size=min_face_size)
            logger.info("Initialized MTCNN on device %s.", device)
    
    def detect(self, frame: np.ndarray):
        """
        frame: BGR numpy array (OpenCV)
        returns list of (x1,y1,x2,y2,score)
        """
        if self.mtcnn is None:
            # graceful fallback: no detections
            return []

        # MTCNN expects RGB images (uint8) or PIL; convert BGR->RGB
        try:
            # If frame is empty or invalid, return empty
            if frame is None or frame.size == 0:
                return []

            rgb = frame[:, :, ::-1]  # BGR -> RGB
            # facenet-pytorch MTCNN.detect accepts numpy array and returns boxes and probs
            boxes, probs = self.mtcnn.detect(rgb)
            results = []
            if boxes is None:
                return []
            # boxes: [[x1, y1, x2, y2], ...], probs: [p1, p2, ...]
            for b, p in zip(boxes, probs):
                x1, y1, x2, y2 = map(int, b.tolist())
                score = float(p) if p is not None else 0.0
                results.append((x1, y1, x2, y2, score))
            return results
        except Exception as e:
            logger.exception("MTCNN detection error: %s", e)
            return []