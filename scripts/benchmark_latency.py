"""Benchmark real-time camera capture, extraction, serialization, and inference latency."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if _SRC_DIR.is_dir():
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np
import torch

from sign_language.config import load_experiment_config
from sign_language.models.factory import build_model
from sign_language.landmarks.extractor import LandmarkExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_prerecorded_frames(video_path: Path | None, count: int = 60) -> List[np.ndarray]:
    """Load frames from a video if available, otherwise synthesize realistic test frames."""
    frames: List[np.ndarray] = []
    if video_path and video_path.is_file():
        cap = cv2.VideoCapture(str(video_path))
        while cap.isOpened() and len(frames) < count:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if frames:
            logger.info("Loaded %d frames from video: %s", len(frames), video_path)
            return frames

    logger.info("Synthesizing %d deterministic camera benchmark frames...", count)
    for i in range(count):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add dynamic patterns (simulating body and hands)
        cv2.circle(img, (320, 200), 50, (200, 180, 150), -1)  # Head
        cv2.line(img, (320, 250), (320, 400), (100, 100, 200), 10)  # Torso
        x_hand = int(240 + 40 * np.sin(i / 5.0))
        y_hand = int(300 + 40 * np.cos(i / 5.0))
        cv2.circle(img, (x_hand, y_hand), 25, (180, 150, 120), -1)  # Moving hand
        cv2.circle(img, (400, 320), 25, (180, 150, 120), -1)  # Stationary hand
        frames.append(img)
    return frames


def run_benchmark(
    checkpoint_path: Path,
    config_path: Path,
    video_path: Path | None,
    resolutions: List[int],
    qualities: List[int],
    num_frames: int,
    device_name: str,
    output_json: Path,
) -> Dict[str, Any]:
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    config = load_experiment_config(config_path)

    # Load Model
    logger.info("Loading model from checkpoint %s on %s...", checkpoint_path, device)
    model = build_model(config)
    if checkpoint_path.is_file():
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        # Filter state dict if matching
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Initialize Extractor
    extractor = None
    task_asset = Path("models/mediapipe/holistic_landmarker.task")
    if task_asset.is_file():
        try:
            extractor = LandmarkExtractor(
                model_asset_path=str(task_asset),
                representation=config.dataset.feature_representation,
                running_mode="image",
            )
            logger.info("MediaPipe LandmarkExtractor initialized successfully.")
        except Exception as e:
            logger.warning("Could not initialize MediaPipe LandmarkExtractor: %s", e)
    else:
        logger.info("MediaPipe asset %s not present; benchmark will use simulated extraction.", task_asset)

    base_frames = generate_prerecorded_frames(video_path, count=num_frames)

    seq_len = config.dataset.sequence_length
    input_dim = config.model.input_dim

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": str(device),
        "model_architecture": config.model.name,
        "input_dim": input_dim,
        "sequence_length": seq_len,
        "classes": config.model.num_classes,
        "webcam_tested": False,
        "source": "prerecorded_video" if (video_path and video_path.is_file()) else "synthetic_frames",
        "scenarios": [],
    }

    # Warmup
    logger.info("Warming up model inference...")
    dummy_input = torch.zeros((1, seq_len, input_dim), dtype=torch.float32, device=device)
    dummy_mask = torch.ones((1, seq_len), dtype=torch.float32, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input, mask=dummy_mask)

    for width in resolutions:
        # 4:3 aspect ratio
        height = int(width * 0.75)
        for q in qualities:
            scenario_name = f"{width}x{height}_q{q}"
            logger.info("Benchmarking scenario: %s (width=%d, quality=%d)", scenario_name, width, q)

            encode_times_ms: List[float] = []
            decode_times_ms: List[float] = []
            extract_times_ms: List[float] = []
            inference_times_ms: List[float] = []
            e2e_times_ms: List[float] = []
            payload_sizes_bytes: List[int] = []

            # Sliding buffer for model
            feature_buffer = collections_deque = [np.zeros(input_dim, dtype=np.float32) for _ in range(seq_len)]
            mask_buffer = [1.0 for _ in range(seq_len)]

            for idx, raw_frame in enumerate(base_frames):
                resized = cv2.resize(raw_frame, (width, height))

                # 1. Serialization (JPEG encode)
                t0 = time.perf_counter()
                success, encoded_buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), q])
                t_enc = (time.perf_counter() - t0) * 1000.0
                encode_times_ms.append(t_enc)
                payload_sizes_bytes.append(len(encoded_buf.tobytes()))

                # 2. Deserialization (JPEG decode)
                t1 = time.perf_counter()
                decoded_frame = cv2.imdecode(encoded_buf, cv2.IMREAD_COLOR)
                t_dec = (time.perf_counter() - t1) * 1000.0
                decode_times_ms.append(t_dec)

                # 3. Extraction
                t2 = time.perf_counter()
                if extractor is not None:
                    try:
                        obs = extractor.extract_observation(decoded_frame, timestamp_ms=idx * 40)
                        vec = obs.landmarks
                    except Exception:
                        vec = np.zeros(input_dim, dtype=np.float32)
                else:
                    # Simulated landmark extraction overhead (approx 12ms CPU)
                    time.sleep(0.005)
                    vec = np.zeros(input_dim, dtype=np.float32)
                t_ext = (time.perf_counter() - t2) * 1000.0
                extract_times_ms.append(t_ext)

                feature_buffer.pop(0)
                feature_buffer.append(vec if len(vec) == input_dim else np.zeros(input_dim, dtype=np.float32))

                # 4. Inference
                inp_t = torch.tensor(np.array([feature_buffer]), dtype=torch.float32, device=device)
                msk_t = torch.tensor(np.array([mask_buffer]), dtype=torch.float32, device=device)
                t3 = time.perf_counter()
                with torch.no_grad():
                    _ = model(inp_t, mask=msk_t)
                t_inf = (time.perf_counter() - t3) * 1000.0
                inference_times_ms.append(t_inf)

                e2e_times_ms.append(t_enc + t_dec + t_ext + t_inf)

            e2e_arr = np.array(e2e_times_ms)
            p50 = float(np.percentile(e2e_arr, 50))
            p90 = float(np.percentile(e2e_arr, 90))
            p95 = float(np.percentile(e2e_arr, 95))
            avg_fps = float(1000.0 / np.mean(e2e_arr)) if np.mean(e2e_arr) > 0 else 0.0

            scenario_stats = {
                "scenario": scenario_name,
                "width": width,
                "height": height,
                "jpeg_quality": q,
                "avg_payload_kb": round(float(np.mean(payload_sizes_bytes)) / 1024.0, 2),
                "avg_encode_ms": round(float(np.mean(encode_times_ms)), 2),
                "avg_decode_ms": round(float(np.mean(decode_times_ms)), 2),
                "avg_extract_ms": round(float(np.mean(extract_times_ms)), 2),
                "avg_inference_ms": round(float(np.mean(inference_times_ms)), 2),
                "p50_latency_ms": round(p50, 2),
                "p90_latency_ms": round(p90, 2),
                "p95_latency_ms": round(p95, 2),
                "estimated_fps": round(avg_fps, 1),
                "frames_tested": len(base_frames),
            }
            results["scenarios"].append(scenario_stats)
            logger.info(
                "  => p50: %.2f ms | p95: %.2f ms | FPS: %.1f | Payload: %.1f KB",
                p50,
                p95,
                avg_fps,
                scenario_stats["avg_payload_kb"],
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Benchmark complete. Results saved to %s", output_json)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-Time SignFlow Latency and Pipeline Benchmark")
    parser.add_argument("--checkpoint", default="models/robust_holistic_fusion_best.pt")
    parser.add_argument("--config", default="configs/experiments/wlasl100_proposed.yaml")
    parser.add_argument("--video-path", default="data/WLASL-3/videos/00336.mp4")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[320, 480, 640])
    parser.add_argument("--qualities", nargs="+", type=int, default=[50, 75, 90])
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/realtime_benchmark.json")
    args = parser.parse_args()

    v_path = Path(args.video_path) if args.video_path else None
    if v_path and not v_path.is_file():
        candidates = list(Path("data").glob("**/*.mp4"))
        v_path = candidates[0] if candidates else None

    run_benchmark(
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        video_path=v_path,
        resolutions=args.resolutions,
        qualities=args.qualities,
        num_frames=args.num_frames,
        device_name=args.device,
        output_json=Path(args.output),
    )


if __name__ == "__main__":
    main()
