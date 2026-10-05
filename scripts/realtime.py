from __future__ import annotations

import argparse
import time
import cv2
import numpy as np

from sign_language.api.inference import RealTimePredictor
from sign_language.landmarks.extractor import LandmarkExtractor
from sign_language.landmarks.pipeline import LandmarkPipeline

def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time sign language recognition.")
    parser.add_argument("--video-source", default="0", help="Camera index or path to video")
    parser.add_argument("--model-path", help="Path to trained PyTorch model (.pt)")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to config")
    parser.add_argument("--model-asset-path", default="models/mediapipe/holistic_landmarker.task")
    args = parser.parse_args()

    predictor = RealTimePredictor(model_path=args.model_path, config_path=args.config)
    extractor = LandmarkExtractor(model_asset_path=args.model_asset_path, representation="holistic")
    pipeline = LandmarkPipeline(feature_dim=predictor.config.model.input_dim, 
                                sequence_length=predictor.config.dataset.sequence_length, 
                                extractor=extractor)

    source = int(args.video_source) if args.video_source.isdigit() else args.video_source
    cap = cv2.VideoCapture(source)

    print("Starting real-time recognition. Press 'q' to quit.")
    
    sequence_buffer = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Extract features for single frame
        observation = extractor.extract_observation(frame)
        sequence_buffer.append(observation.landmarks)
        
        # Keep sequence at desired length
        if len(sequence_buffer) > pipeline.sequence_length:
            sequence_buffer.pop(0)
            
        if len(sequence_buffer) == pipeline.sequence_length:
            seq_array = np.stack(sequence_buffer)
            # Normalize and add dynamics
            norm_seq = pipeline.normalize_sequence(seq_array)
            logits = predictor.predict(norm_seq)
            pred_class = int(np.argmax(logits))
            
            # Display prediction
            if predictor.config.dataset.labels and pred_class < len(predictor.config.dataset.labels):
                label = predictor.config.dataset.labels[pred_class]
            else:
                label = f"Class {pred_class}"
                
            cv2.putText(frame, f"Pred: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        cv2.imshow("Sign Language Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
