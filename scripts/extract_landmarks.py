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

import multiprocessing
from functools import partial

def process_video_task(entry: dict, raw_dir: Path, output_dir: Path, model_asset_path: str, representation: str) -> tuple[bool, str, str]:
    sample_id = entry["sample_id"]
    # The entry["video_path"] in manifest could be just the class_name/filename or similar.
    # WLASL manifest videos are already relative to raw_dir if we generated them that way,
    # or just raw paths.
    video_path = raw_dir / entry["video_path"]
    output_path = output_dir / f"{sample_id}.npz"
    
    if output_path.exists():
        return True, sample_id, ""

    try:
        extractor = LandmarkExtractor(
            model_asset_path=model_asset_path,
            representation=representation
        )
        pipeline = LandmarkPipeline(extractor=extractor)
        
        seq = pipeline.process_video(str(video_path))
        np.savez_compressed(output_path, landmarks=seq.landmarks, mask=seq.mask)
        return True, sample_id, ""
    except Exception as e:
        return False, sample_id, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract landmarks for all videos in a manifest.")
    parser.add_argument("--manifest", required=True, help="Path to input manifest json")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw videos")
    parser.add_argument("--output-dir", default="data/landmarks", help="Directory to save extracted landmarks")
    parser.add_argument("--model-asset-path", default="models/mediapipe/holistic_landmarker.task")
    parser.add_argument("--representation", default="holistic")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of workers")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    logger.info(f"Loaded {len(manifest)} entries from {args.manifest}. Using {args.workers} workers.")

    success = 0
    failed = 0

    task_func = partial(
        process_video_task,
        raw_dir=raw_dir,
        output_dir=output_dir,
        model_asset_path=args.model_asset_path,
        representation=args.representation
    )

    with multiprocessing.Pool(processes=args.workers) as pool:
        for i, (is_success, sample_id, err_msg) in enumerate(pool.imap_unordered(task_func, manifest)):
            if is_success:
                success += 1
            else:
                failed += 1
                logger.warning(f"Failed to process {sample_id}: {err_msg}")
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processing {i + 1}/{len(manifest)}... (Success: {success}, Failed: {failed})")

    logger.info(f"Extraction complete. Success: {success}, Failed: {failed}")

if __name__ == "__main__":
    main()
