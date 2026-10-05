from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from sign_language.landmarks.extractor import LandmarkExtractor
from sign_language.landmarks.pipeline import LandmarkPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract landmarks for all videos in a manifest.")
    parser.add_argument("--manifest", required=True, help="Path to input manifest json")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw videos")
    parser.add_argument("--output-dir", default="data/landmarks", help="Directory to save extracted landmarks")
    parser.add_argument("--model-asset-path", default="models/mediapipe/holistic_landmarker.task")
    parser.add_argument("--representation", default="holistic")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    logger.info(f"Loaded {len(manifest)} entries from {args.manifest}")

    success = 0
    failed = 0

    for idx, entry in enumerate(manifest):
        sample_id = entry["sample_id"]
        video_path = raw_dir / entry["video_path"]
        output_path = output_dir / f"{sample_id}.npz"
        
        if output_path.exists():
            success += 1
            continue

        if idx % 100 == 0:
            logger.info(f"Processing {idx}/{len(manifest)}...")

        try:
            # Recreate extractor and pipeline per video to clear MediaPipe's internal segmentation state.
            # Otherwise, processing videos of different resolutions sequentially causes a SegmentationSmoothingCalculator crash.
            extractor = LandmarkExtractor(
                model_asset_path=args.model_asset_path,
                representation=args.representation
            )
            pipeline = LandmarkPipeline(extractor=extractor)
            
            seq = pipeline.process_video(str(video_path))
            np.savez_compressed(output_path, landmarks=seq.landmarks, mask=seq.mask)
            success += 1
        except Exception as e:
            logger.warning(f"Failed to process {sample_id}: {e}")
            failed += 1

    logger.info(f"Extraction complete. Success: {success}, Failed: {failed}")

if __name__ == "__main__":
    main()
