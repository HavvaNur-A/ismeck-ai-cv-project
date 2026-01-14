# utils/viz.py
import cv2
from typing import List, Tuple


# tracks: List[(id, (x1,y1,x2,y2))]
def draw_boxes_with_ids(frame, tracks: List[Tuple[int, Tuple[int,int,int,int]]]):
    for tid, bb in tracks:
        x1, y1, x2, y2 = map(int, bb)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame

def put_probabilities(frame, ids, tracks, probs):
    # assumes ids aligned with probs and tracks order consistent
    id_to_bb = {tid: bb for tid, bb in tracks}
    for tid, p in zip(ids, probs):
        bb = id_to_bb.get(tid)
        if bb is None:
            continue
        x1, y1, x2, y2 = map(int, bb)
        cv2.putText(frame, f"{p:.2f}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    return frame