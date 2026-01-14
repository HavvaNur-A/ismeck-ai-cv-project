# pipeline/pipeline.py
import cv2
from typing import Any, List, Tuple
from detectors.base import BaseDetector
from trackers.base import BaseTracker
from models.base import BaseModelWrapper
from utils.viz import draw_boxes_with_ids, put_probabilities
import numpy as np
from config import logger



class Pipeline:
    def __init__(self, detector: BaseDetector, tracker: BaseTracker, model_wrapper: BaseModelWrapper, visualize: bool = True):
        self.detector = detector
        self.tracker = tracker
        self.model = model_wrapper
        self.visualize = visualize
    
    def run_stream(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error("Cannot open source %s", source)
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dets = self.detector.detect(frame)
            tracks = self.tracker.update(dets)
            crops = []
            ids = []
            for tid, bb in tracks:
                x1, y1, x2, y2 = map(int, bb)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1]-1, x2)
                y2 = min(frame.shape[0]-1, y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    # fallback dummy crop
                    crop = np.zeros((224,224,3), dtype=np.uint8)
                crops.append(crop)
                ids.append(tid)
            probs = []
            if crops:
                try:
                    probs = self.model.infer_batch(crops)
                except Exception as e:
                    logger.exception("Model inference failed: %s", e)
                    probs = [0.0] * len(crops)
            # visualize
            out = frame.copy()
            out = draw_boxes_with_ids(out, tracks)
            out = put_probabilities(out, ids, tracks, probs)
            if self.visualize:
                cv2.imshow('pipeline', out)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
