import argparse
from config import logger
from pipeline.pipeline import Pipeline
from detectors.mtcnn_detector import MTCNNDetector
from trackers.iou_tracker import IOUTracker
from models.vit_wrapper import ViTModelWrapper
import sys


def build_components(detector_name="mtcnn", tracker_name="iou", model_name="vit"):
    if detector_name == "mtcnn":
        detector = MTCNNDetector()
    else: 
        raise ValueError("Unknown Detector")
    
    if tracker_name == "iou":
        tracker = IOUTracker()
    else: 
        raise ValueError("Unknown Tracker")
    
    if model_name == "vit":
        model = ViTModelWrapper()
    else: 
        raise ValueError("Unknown Model")
    
    return detector, tracker, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=0, help="video file or camera comm port index")
    parser.add_argument("--detector", type=str, default="mtcnn")
    parser.add_argument("--tracker", type=str, default="iou")
    parser.add_argument("--model", type=str, default="vit")
    args = parser.parse_args()

    #src = int(args.source) if args.source.isdigit() else args.source    
    src = int(str(args.source)) if str(args.source).isdigit() else str(args.source)
    det, trk, mdl = build_components(args.detector, args.tracker, args.model)
    pipeline = Pipeline(detector=det, tracker=trk, model_wrapper=mdl, visualize=True)
    logger.info("Starting pipeline with detector=%s, tracker=%s, model=%s.", args.detector, args.tracker, args.model)
    pipeline.run_stream(src)

if __name__ == "__main__":
    main()