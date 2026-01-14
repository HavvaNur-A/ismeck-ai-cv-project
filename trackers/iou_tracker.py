from .base import BaseTracker, Track
from typing import List, Tuple
import math



class IOUTracker(BaseTracker):
    def __init__(self, iou_tresh: float=0.3):
        self.tracks = {} # id -> bbox | OCR, Segmentasyon, Color based Clustering vb..
        self.next_id = 0
        self.iou_tresh = iou_tresh

    @staticmethod
    def iou(a, b):
        # a -> (x1_a, y1_a, x2_a, y2_a)
        xA = max(a[0], b[0]) # sol kenar
        yA = min(a[1], b[1]) # üst kenar
        # b -> (x1_b, y1_b, x2_b, y2_b)
        xB = max(a[2], b[2]) # sağ kenar
        yB = min(a[3], b[3]) # alt kenar
        
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        inter = interW * interH

        areaA = max(0, (a[2] - a[0])) * max(0, (a[3] - a[1])) # ilk frame
        areaB = max(0, (b[2] - b[0])) * max(0, (b[3] - b[1])) # ikinci frame

        union = areaA + areaB - inter
        return inter / union if union > 0 else 0.0
    
    def update(self, detections: List[Tuple[int,int,int,int,float]]) -> List[Track]:
        assigned = set()
        new_tracks = {}

        for tid, tb in list(self.tracks.items()):
            best_iou = 0.0 
            best_idx = None
            for idx, det in enumerate(detections):
                if idx in assigned:
                    continue
                i = self.iou(tb, det[:4])
                if i > best_iou:
                    best_iou = i
                    best_idx = idx
            if best_idx is not None and best_iou >= self.iou_tresh:
                new_tracks[tid] = detections[best_idx][:4]
                assigned.add(best_idx)
        
        # create task for unassigned detections
        for idx, det in enumerate(detections):
            if idx in assigned:
                continue
            new_tracks[self.next_id] = det[:4]
            self.next_id += 1
        
        self.tracks = new_tracks
        return [
            (tid, tuple(map(int, bb))) 
            for tid, bb in self.tracks.items()]