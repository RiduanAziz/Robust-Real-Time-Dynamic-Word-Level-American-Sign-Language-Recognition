from __future__ import annotations

import base64
import collections
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sign_language.api.inference import RealTimePredictor, create_prediction_payload
from sign_language.landmarks.extractor import LandmarkExtractor
from sign_language.landmarks.pipeline import resample_sequence
from sign_language.robustness import apply_spatial_noise, apply_temporal_noise

logger = logging.getLogger(__name__)

# Global singleton predictor and extractor instances
_predictor: RealTimePredictor | None = None
_extractor: LandmarkExtractor | None = None
_init_error: str | None = None
_extractor_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _extractor, _init_error, _extractor_error
    model_path = os.environ.get("MODEL_PATH")
    config_path = os.environ.get("CONFIG_PATH")
    mediapipe_asset = os.environ.get(
        "MEDIAPIPE_ASSET_PATH", "models/mediapipe/holistic_landmarker.task"
    )

    # Prioritize user-trained temporal_transformer_trained.pt
    if not model_path:
        candidates = [
            ("models/temporal_transformer_trained.pt", "configs/base.yaml"),
            ("models/temporal_transformer_best.pt", "configs/base.yaml"),
            ("models/robust_holistic_fusion_best.pt", "configs/experiments/wlasl20_proposed.yaml"),
        ]
        for m_cand, c_cand in candidates:
            if Path(m_cand).is_file():
                model_path = m_cand
                if not config_path and Path(c_cand).is_file():
                    config_path = c_cand
                break

    if not config_path:
        config_path = "configs/base.yaml"

    # Initialize RealTimePredictor strictly from trained checkpoint
    try:
        if model_path and Path(model_path).is_file():
            _predictor = RealTimePredictor(model_path=model_path, config_path=config_path)
            logger.info("RealTimePredictor successfully initialized with model %s", model_path)
            _init_error = None
        else:
            _predictor = None
            _init_error = (
                f"No trained model checkpoint found at {model_path or 'models/temporal_transformer_trained.pt'}. "
                "Recognition disabled until valid checkpoint is supplied."
            )
            logger.warning(_init_error)
    except Exception as exc:
        _predictor = None
        _init_error = str(exc)
        logger.warning("Failed to initialize predictor at startup: %s", exc)

    # Initialize LandmarkExtractor
    task_path = Path(mediapipe_asset)
    if task_path.is_file():
        try:
            rep = _predictor.config.dataset.feature_representation if _predictor else "holistic"
            _extractor = LandmarkExtractor(
                model_asset_path=task_path,
                representation=rep,
                running_mode="image",
            )
            _extractor_error = None
            logger.info("LandmarkExtractor successfully initialized with %s", task_path)
        except Exception as exc:
            _extractor_error = str(exc)
            logger.warning("Failed to initialize LandmarkExtractor: %s", exc)
    else:
        _extractor_error = f"MediaPipe asset not found at {task_path}"
        logger.info("LandmarkExtractor disabled: %s", _extractor_error)

    yield

    # Clean up on shutdown
    if _extractor is not None:
        try:
            _extractor.close()
        except Exception as exc:
            logger.debug("Error closing extractor on shutdown: %s", exc)


app = FastAPI(
    title="SignFlow — Live ASL Recognition API",
    description="Backend API and WebSocket service for real-time American Sign Language recognition.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_predictor() -> RealTimePredictor:
    global _predictor, _init_error
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Predictor service unavailable: {_init_error or 'No trained model checkpoint loaded.'}",
        )
    return _predictor


def get_extractor() -> LandmarkExtractor | None:
    global _extractor
    return _extractor


class PredictionRequest(BaseModel):
    sequence: list[list[float]] = Field(..., description="2D sequence of landmark features [T, F]")
    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum confidence threshold")


@app.get("/health")
def health() -> dict[str, Any]:
    mediapipe_ready = _extractor is not None and getattr(_extractor, "_landmarker", None) is not None
    if _extractor is not None and not mediapipe_ready:
        # Check if model asset file exists even if landmarker not yet lazily created
        mediapipe_ready = (
            _extractor.model_asset_path is not None and _extractor.model_asset_path.is_file()
        )

    model_ready = (
        _predictor is not None
        and getattr(_predictor, "model", None) is not None
        and _init_error is None
    )

    if model_ready and mediapipe_ready:
        overall_status = "ok"
    elif model_ready or mediapipe_ready:
        overall_status = "degraded"
    else:
        overall_status = "error"

    return {
        "status": overall_status,
        "process_alive": True,
        "backend_initialized": True,
        "mediapipe_available": mediapipe_ready,
        "trained_model_loaded": model_ready,
        "inference_ready": model_ready,
        "checkpoint_path": getattr(_predictor, "checkpoint_path", None) if _predictor else None,
        "error": _init_error,
        "mediapipe_error": _extractor_error,
    }


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    predictor = get_predictor()
    return {
        "name": predictor.config.model.name,
        "architecture": predictor.model.__class__.__name__ if hasattr(predictor, "model") else "unknown",
        "status": "ready" if _init_error is None else "error",
        "checkpoint_path": getattr(predictor, "checkpoint_path", None),
        "num_classes": predictor.num_classes,
        "input_dim": predictor.input_dim,
        "sequence_length": predictor.sequence_length,
        "device": str(predictor.device),
        "feature_representation": predictor.config.dataset.feature_representation,
        "class_names_sample": predictor.class_names[:10] if predictor.class_names else [],
        "total_classes": len(predictor.class_names),
        "error": _init_error,
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if not request.sequence:
        raise HTTPException(status_code=422, detail="Sequence must not be empty.")

    try:
        sequence = np.asarray(request.sequence, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid numeric sequence data: {exc}")

    if sequence.ndim != 2:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 2D array [time_steps, features], got {sequence.shape}",
        )

    if not np.all(np.isfinite(sequence)):
        raise HTTPException(status_code=422, detail="Input sequence contains non-finite values (NaN or Inf).")

    predictor = get_predictor()

    try:
        prediction_result = predictor.predict_label(
            sequence, confidence_threshold=request.confidence_threshold
        )
        logits = predictor.predict_logits(sequence)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    payload = create_prediction_payload(sequence)

    return {
        "sequence_length": payload["sequence_length"],
        "feature_dim": payload["feature_dim"],
        "logits": [float(v) for v in logits],
        "predicted_class": prediction_result["predicted_class"],
        "predicted_label": prediction_result["predicted_label"],
        "confidence": prediction_result["confidence"],
        "meets_threshold": prediction_result["meets_threshold"],
        "top_k": prediction_result.get("top_k", []),
    }


class RobustnessExperimentRequest(BaseModel):
    sequence: list[list[float]] = Field(..., description="Sequence of landmark features [T, F]")
    perturbation_type: str = Field(
        ...,
        description="Perturbation type (coordinate_jitter, translation, scale, landmark_dropout, frame_drop, frame_duplicate, sequence_truncate)",
    )
    severity: float = Field(0.2, ge=0.0, le=1.0, description="Perturbation severity")
    seed: int = Field(42, description="Random seed")


@app.get("/api/vocabulary")
def get_vocabulary() -> dict[str, Any]:
    """Returns the true vocabulary of the loaded model checkpoint for Guided Practice Mode."""
    predictor = _predictor
    if predictor is None:
        return {
            "available": False,
            "error": _init_error or "Model checkpoint not loaded.",
            "vocabulary": [],
            "count": 0,
            "checkpoint": None,
        }
    return {
        "available": True,
        "vocabulary": predictor.class_names,
        "count": len(predictor.class_names),
        "checkpoint": getattr(predictor, "checkpoint_path", None),
    }


@app.post("/api/robustness/experiment")
def run_robustness_experiment(request: RobustnessExperimentRequest) -> dict[str, Any]:
    """Thesis-aligned robustness analysis comparing clean vs perturbed inference."""
    predictor = get_predictor()
    try:
        seq_clean = np.asarray(request.sequence, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid numeric sequence data: {exc}")

    if seq_clean.ndim != 2:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 2D array [T, F], got shape {seq_clean.shape}",
        )

    t0 = time.perf_counter()
    clean_pred = predictor.predict_label(seq_clean, confidence_threshold=0.0)

    noise_type = request.perturbation_type
    severity = float(request.severity)
    seed = int(request.seed)

    spatial_types = {"coordinate_jitter", "translation", "scale", "landmark_dropout"}
    temporal_types = {"frame_drop", "frame_duplicate", "sequence_truncate"}

    try:
        if noise_type in spatial_types:
            perturbed_seq = apply_spatial_noise(seq_clean, noise_type, severity=severity, seed=seed)
        elif noise_type in temporal_types:
            result = apply_temporal_noise(seq_clean, noise_type, severity=severity, seed=seed)
            perturbed_seq = result[0] if isinstance(result, tuple) else result
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported perturbation '{noise_type}'. Must be one of {sorted(spatial_types | temporal_types)}",
            )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Perturbation execution error: {exc}")

    try:
        perturbed_pred = predictor.predict_label(perturbed_seq, confidence_threshold=0.0)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Perturbed prediction execution error: {exc}")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    is_consistent = bool(clean_pred["predicted_label"] == perturbed_pred["predicted_label"])
    clean_sparsity = float(1.0 - (np.count_nonzero(seq_clean) / max(1, seq_clean.size)))
    perturbed_sparsity = float(1.0 - (np.count_nonzero(perturbed_seq) / max(1, perturbed_seq.size)))

    return {
        "original_prediction": clean_pred["predicted_label"],
        "original_confidence": float(clean_pred["confidence"]),
        "perturbed_prediction": perturbed_pred["predicted_label"],
        "perturbed_confidence": float(perturbed_pred["confidence"]),
        "prediction_consistent": is_consistent,
        "prediction_changed": not is_consistent,
        "perturbation_type": noise_type,
        "severity": severity,
        "processing_time_ms": round(elapsed_ms, 2),
        "clean_quality": {
            "sequence_length": int(seq_clean.shape[0]),
            "feature_dim": int(seq_clean.shape[1]),
            "sparsity_ratio": round(clean_sparsity, 3),
        },
        "perturbed_quality": {
            "sequence_length": int(perturbed_seq.shape[0]),
            "feature_dim": int(perturbed_seq.shape[1]),
            "sparsity_ratio": round(perturbed_sparsity, 3),
        },
        "note": "Prediction consistency is reported for this live sequence. Ground-truth accuracy is evaluated on labelled offline benchmark datasets.",
    }


@app.websocket("/ws/live")
async def websocket_live_recognition(websocket: WebSocket) -> None:
    """Persistent bidirectional WebSocket connection for real-time ASL recognition."""
    await websocket.accept()

    session_id = str(uuid.uuid4())
    logger.info("WebSocket client connected. Session ID: %s", session_id)

    predictor = _predictor
    extractor = _extractor

    # Check initial availability
    model_ready = predictor is not None and getattr(predictor, "model", None) is not None
    extractor_ready = extractor is not None

    seq_len = predictor.sequence_length if predictor else 64
    raw_dim = (
        predictor.input_dim // 3
        if (predictor and predictor.input_dim >= 4977)
        else (predictor.input_dim if predictor else 1659)
    )

    # Initial session handshake
    await websocket.send_json({
        "type": "session_started",
        "session_id": session_id,
        "model_ready": model_ready,
        "model_name": predictor.config.model.name if predictor else None,
        "num_classes": predictor.num_classes if predictor else 0,
        "sequence_length": seq_len,
        "feature_dim": raw_dim,
        "mediapipe_ready": extractor_ready,
        "active_mode": "guided",
        "supported_modes": ["guided", "continuous"],
    })

    # Connection-level state (tuned for dynamic motion stroke recognition)
    mode = "guided"  # "guided" or "continuous"
    confidence_threshold = 0.28
    debounce_frames = 2
    pause_threshold_sec = 0.5
    min_sequence_frames = 10

    sequence_buffer: collections.deque = collections.deque(maxlen=seq_len)
    mask_buffer: collections.deque = collections.deque(maxlen=seq_len)
    candidate_history: collections.deque = collections.deque(maxlen=debounce_frames)
    recent_replay_frames: collections.deque = collections.deque(maxlen=64)
    wrist_history: collections.deque = collections.deque(maxlen=5)
    speed_history: collections.deque = collections.deque(maxlen=10)

    last_brightness: float | None = None
    last_blur_score: float | None = None

    current_candidate: str | None = None
    current_confidence: float = 0.0
    last_committed_word: str | None = None
    state: str = "WAITING"  # "WAITING" | "CAPTURING" | "CANDIDATE" | "COMMITTED"

    last_hand_activity_time = time.time()
    frame_counter = 0
    start_time = time.time()

    try:
        while True:
            # Handle text or binary messages
            message = await websocket.receive()

            if "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON format."})
                    continue

                msg_type = payload.get("type", "")

                # Replay buffer request
                if msg_type == "request_replay":
                    await websocket.send_json({
                        "type": "replay_buffer",
                        "frames": list(recent_replay_frames),
                    })
                    continue

                # 1. Configuration update
                if msg_type == "config":
                    if "mode" in payload and payload["mode"] in {"guided", "continuous"}:
                        mode = payload["mode"]
                    if "confidence_threshold" in payload:
                        confidence_threshold = float(payload["confidence_threshold"])
                    if "debounce_frames" in payload:
                        debounce_frames = max(2, int(payload["debounce_frames"]))
                        candidate_history = collections.deque(
                            list(candidate_history), maxlen=debounce_frames
                        )
                    if "pause_threshold_sec" in payload:
                        pause_threshold_sec = float(payload["pause_threshold_sec"])
                    await websocket.send_json({
                        "type": "config_updated",
                        "mode": mode,
                        "confidence_threshold": confidence_threshold,
                        "debounce_frames": debounce_frames,
                        "pause_threshold_sec": pause_threshold_sec,
                    })
                    continue

                # 2. Explicit manual confirm word
                elif msg_type == "confirm_word":
                    if current_candidate:
                        token_id = str(uuid.uuid4())[:8]
                        await websocket.send_json({
                            "type": "word_committed",
                            "token_id": token_id,
                            "word": current_candidate,
                            "confidence": current_confidence,
                            "timestamp": time.time(),
                            "mode": mode,
                            "source": "manual_confirm",
                        })
                        last_committed_word = current_candidate
                        state = "COMMITTED"
                        current_candidate = None
                        sequence_buffer.clear()
                        candidate_history.clear()
                    continue

                # 3. Explicit reject candidate
                elif msg_type == "reject_candidate":
                    current_candidate = None
                    current_confidence = 0.0
                    candidate_history.clear()
                    state = "WAITING"
                    await websocket.send_json({
                        "type": "word_candidate",
                        "word": None,
                        "confidence": 0.0,
                        "stability_count": 0,
                        "target_count": debounce_frames,
                        "progress": 0.0,
                    })
                    continue

                # 4. Reset session buffers
                elif msg_type == "reset":
                    sequence_buffer.clear()
                    mask_buffer.clear()
                    candidate_history.clear()
                    wrist_history.clear()
                    speed_history.clear()
                    current_candidate = None
                    last_committed_word = None
                    state = "WAITING"
                    await websocket.send_json({"type": "reset_ack"})
                    continue

                # 5. Process Base64 video frame
                elif msg_type == "frame":
                    frame_data = payload.get("data", "")
                    ts_ms = payload.get("timestamp_ms", int(time.time() * 1000))
                    if not frame_data:
                        continue

                    # Strip data URL header if present
                    if "," in frame_data:
                        frame_data = frame_data.split(",", 1)[1]

                    try:
                        img_bytes = base64.b64decode(frame_data)
                        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    except Exception as exc:
                        await websocket.send_json({"type": "warning", "message": f"Frame decode error: {exc}"})
                        continue

                    if frame is None:
                        continue

                else:
                    continue

            elif "bytes" in message and message["bytes"]:
                # Binary JPEG frame
                ts_ms = int(time.time() * 1000)
                try:
                    img_arr = np.frombuffer(message["bytes"], dtype=np.uint8)
                    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                except Exception as exc:
                    await websocket.send_json({"type": "warning", "message": f"Binary decode error: {exc}"})
                    continue

                if frame is None:
                    continue

            else:
                continue

            # Process valid frame
            frame_counter += 1
            now = time.time()

            # Ensure extractor is ready
            if extractor is None:
                await websocket.send_json({
                    "type": "landmarks_status",
                    "hands_detected": False,
                    "left_hand": False,
                    "right_hand": False,
                    "pose": False,
                    "face": False,
                    "valid_points": 0,
                    "error": "MediaPipe landmark extractor unavailable.",
                })
                continue

            # 1. Landmark extraction (Single high-speed pass)
            t0 = time.perf_counter()
            try:
                structured = extractor.extract_structured_frame(frame, timestamp_ms=int(ts_ms))
            except Exception as exc:
                logger.debug("Extraction failed for frame: %s", exc)
                continue
            extraction_ms = (time.perf_counter() - t0) * 1000.0

            # Extract observation features directly from structured frame (no duplicate detect call)
            selected_modalities = ["left_hand", "right_hand"]
            if extractor.representation in {"hands_pose", "holistic"}:
                selected_modalities.append("pose")
            if extractor.representation == "holistic":
                selected_modalities.append("face")

            mod_values: list[np.ndarray] = []
            mod_masks: list[np.ndarray] = []
            for m_name in selected_modalities:
                c_mod, m_mod = structured.get_modality(m_name)
                mod_values.append(c_mod.reshape(-1))
                mod_masks.append(np.repeat(m_mod, 3).astype(np.float32))

            obs_landmarks = np.concatenate(mod_values).astype(np.float32)
            obs_mask = np.concatenate(mod_masks).astype(np.float32)
            if obs_landmarks.shape[0] > extractor.feature_dim:
                obs_landmarks = obs_landmarks[: extractor.feature_dim]
                obs_mask = obs_mask[: extractor.feature_dim]
            elif obs_landmarks.shape[0] < extractor.feature_dim:
                obs_landmarks = np.pad(obs_landmarks, (0, extractor.feature_dim - obs_landmarks.shape[0]))
                obs_mask = np.pad(obs_mask, (0, extractor.feature_dim - obs_mask.shape[0]))

            # Compute landmark presence and 21-point coordinates
            coords = structured.coordinates
            mask = structured.point_mask

            left_hand = bool(mask[:21].sum() > 0)
            right_hand = bool(mask[21:42].sum() > 0)
            pose = bool(mask[42:75].sum() > 0)
            face = bool(mask[75:].sum() > 0)
            hands_present = left_hand or right_hand
            valid_points = int(mask.sum())
            num_hands = (1 if left_hand else 0) + (1 if right_hand else 0)

            left_pts = [
                {
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                    "z": float(coords[i, 2]),
                    "valid": bool(mask[i] > 0),
                }
                for i in range(21)
            ]
            right_pts = [
                {
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                    "z": float(coords[i, 2]),
                    "valid": bool(mask[i] > 0),
                }
                for i in range(21, 42)
            ]

            left_box = None
            if left_hand:
                valid_lh = mask[:21] > 0
                l_xs = coords[:21, 0][valid_lh]
                l_ys = coords[:21, 1][valid_lh]
                xmin, xmax = float(l_xs.min()), float(l_xs.max())
                ymin, ymax = float(l_ys.min()), float(l_ys.max())
                pad = 0.05
                left_box = {
                    "xmin": max(0.0, xmin - pad),
                    "ymin": max(0.0, ymin - pad),
                    "xmax": min(1.0, xmax + pad),
                    "ymax": min(1.0, ymax + pad),
                    "width": min(1.0, max(0.01, (xmax - xmin) + 2 * pad)),
                    "height": min(1.0, max(0.01, (ymax - ymin) + 2 * pad)),
                }

            right_box = None
            if right_hand:
                valid_rh = mask[21:42] > 0
                r_xs = coords[21:42, 0][valid_rh]
                r_ys = coords[21:42, 1][valid_rh]
                xmin, xmax = float(r_xs.min()), float(r_xs.max())
                ymin, ymax = float(r_ys.min()), float(r_ys.max())
                pad = 0.05
                right_box = {
                    "xmin": max(0.0, xmin - pad),
                    "ymin": max(0.0, ymin - pad),
                    "xmax": min(1.0, xmax + pad),
                    "ymax": min(1.0, ymax + pad),
                    "width": min(1.0, max(0.01, (xmax - xmin) + 2 * pad)),
                    "height": min(1.0, max(0.01, (ymax - ymin) + 2 * pad)),
                }

            # Quality Coach metric calculations
            feedback_list = []
            coach_score = 100
            max_box_area = 0.0
            for b in [left_box, right_box]:
                if b:
                    max_box_area = max(max_box_area, b["width"] * b["height"])

            if not hands_present:
                coach_score -= 60
                feedback_list.append("No hand detected. Bring your hand into the camera view.")
            else:
                if max_box_area < 0.03:
                    coach_score -= 20
                    feedback_list.append("Hand is too small. Move your hand closer to the camera.")
                elif max_box_area > 0.65:
                    coach_score -= 15
                    feedback_list.append("Hand is clipped or too close. Move slightly farther away.")

                # Check clipping of tips (4, 8, 12, 16, 20) or wrist (0)
                is_clipped = False
                for pts in [left_pts, right_pts]:
                    for tip_idx in [0, 4, 8, 12, 16, 20]:
                        p = pts[tip_idx]
                        if p["valid"] and (p["x"] < 0.03 or p["x"] > 0.97 or p["y"] < 0.03 or p["y"] > 0.97):
                            is_clipped = True
                            break
                if is_clipped:
                    coach_score -= 20
                    feedback_list.append("Fingertips or wrist are outside/near the frame boundary.")

                # Check centering
                centers = []
                if left_box:
                    centers.append((left_box["xmin"] + left_box["width"] / 2, left_box["ymin"] + left_box["height"] / 2))
                if right_box:
                    centers.append((right_box["xmin"] + right_box["width"] / 2, right_box["ymin"] + right_box["height"] / 2))
                if centers:
                    avg_cx = sum(c[0] for c in centers) / len(centers)
                    avg_cy = sum(c[1] for c in centers) / len(centers)
                    if avg_cx < 0.20 or avg_cx > 0.80 or avg_cy < 0.15 or avg_cy > 0.85:
                        coach_score -= 15
                        feedback_list.append("Move your hand into the center camera guide.")

            # Subsampled thumbnail diagnostics (runs in <0.2ms every 4 frames)
            if frame_counter % 4 == 0 or last_brightness is None:
                thumb = frame[::4, ::4]
                last_brightness = float(np.mean(thumb))
                try:
                    gray_thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
                    last_blur_score = float(cv2.Laplacian(gray_thumb, cv2.CV_64F).var())
                except Exception:
                    last_blur_score = 50.0

            brightness = last_brightness or 120.0
            blur_score = last_blur_score or 50.0

            if brightness < 60:
                coach_score -= 20
                feedback_list.append("Lighting is low; move to a brighter area.")
            elif brightness > 220:
                coach_score -= 15
                feedback_list.append("Lighting is overexposed; reduce direct glare.")

            if blur_score < 25.0 and hands_present:
                coach_score -= 15
                feedback_list.append("Camera image may be blurred. Hold still.")

            stability_val = 1.0
            if hands_present:
                wrist_pos = None
                if right_hand and right_pts[0]["valid"]:
                    wrist_pos = (right_pts[0]["x"], right_pts[0]["y"])
                elif left_hand and left_pts[0]["valid"]:
                    wrist_pos = (left_pts[0]["x"], left_pts[0]["y"])
                if wrist_pos:
                    wrist_history.append(wrist_pos)
                    if len(wrist_history) >= 3:
                        dist = sum(
                            np.hypot(wrist_history[k][0] - wrist_history[k - 1][0], wrist_history[k][1] - wrist_history[k - 1][1])
                            for k in range(1, len(wrist_history))
                        ) / (len(wrist_history) - 1)
                        if dist > 0.12:
                            stability_val = max(0.2, 1.0 - dist * 3)
                            coach_score -= 15
                            feedback_list.append("Landmark tracking is unstable. Hold sign steadily.")

            buf_completeness = min(1.0, len(sequence_buffer) / max(1, min_sequence_frames))
            if hands_present and buf_completeness < 1.0:
                feedback_list.append(f"Collecting sign: {int(buf_completeness * 100)}% of motion window.")

            if not feedback_list and hands_present:
                feedback_list.append("Hand position and lighting are good.")

            coach_score = max(5, min(100, coach_score))

            # Determine recognition state machine code
            if predictor is None:
                state_code = "STATE_G_MODEL_UNAVAILABLE"
            elif not hands_present:
                state_code = "STATE_B_NO_HAND"
            elif len(sequence_buffer) < min_sequence_frames:
                state_code = "STATE_C_COLLECTING"
            else:
                if state == "COMMITTED":
                    state_code = "STATE_E_COMMITTED"
                elif current_candidate and current_candidate not in {"Unknown (Low Confidence)", "Model Unavailable"}:
                    state_code = "STATE_D_CANDIDATE"
                else:
                    state_code = "STATE_F_UNCERTAIN"

            quality_payload = {
                "score": coach_score,
                "level": "Good" if coach_score >= 70 else "Needs Improvement",
                "feedback": feedback_list,
                "num_hands": num_hands,
                "hand_size": round(max_box_area, 3),
                "brightness": round(brightness, 1),
                "blur_score": round(blur_score, 1),
                "stability": round(stability_val, 2),
                "buffer_completeness": round(buf_completeness, 2),
            }

            # Send legacy landmarks_status
            await websocket.send_json({
                "type": "landmarks_status",
                "hands_detected": hands_present,
                "left_hand": left_hand,
                "right_hand": right_hand,
                "pose": pose,
                "face": face,
                "valid_points": valid_points,
            })

            # Send detailed hand_landmarks with 21-points and Quality Coach
            await websocket.send_json({
                "type": "hand_landmarks",
                "left_landmarks": left_pts,
                "right_landmarks": right_pts,
                "left_box": left_box,
                "right_box": right_box,
                "hands_detected": hands_present,
                "left_hand": left_hand,
                "right_hand": right_hand,
                "pose": pose,
                "face": face,
                "valid_points": valid_points,
                "quality": quality_payload,
                "state_code": state_code,
                "timestamp_ms": ts_ms,
            })

            # Append to bounded replay buffer
            recent_replay_frames.append({
                "frame_idx": frame_counter,
                "timestamp_ms": ts_ms,
                "left_landmarks": left_pts,
                "right_landmarks": right_pts,
                "left_box": left_box,
                "right_box": right_box,
                "hands_detected": hands_present,
                "valid_points": valid_points,
                "state_code": state_code,
                "candidate": current_candidate,
                "confidence": current_confidence,
                "features": obs_landmarks.tolist(),
            })

            # Hand activity tracking for neutral pause detection
            if hands_present:
                last_hand_activity_time = now
                if state == "WAITING":
                    state = "CAPTURING"
                # Append to rolling sequence and mask buffer ONLY when hands are actively present
                sequence_buffer.append(obs_landmarks)
                mask_buffer.append(obs_mask)
            else:
                if (now - last_hand_activity_time) > pause_threshold_sec:
                    # Neutral pause reached: cleanly reset gesture boundary buffers
                    if state == "COMMITTED":
                        state = "WAITING"
                        last_committed_word = None  # Allow repeating the same word on next gesture
                        sequence_buffer.clear()
                        mask_buffer.clear()
                        wrist_history.clear()
                        speed_history.clear()
                    elif state == "CAPTURING":
                        state = "WAITING"
                        sequence_buffer.clear()
                        mask_buffer.clear()
                        wrist_history.clear()
                        speed_history.clear()

            # Ensure model is ready
            if predictor is None:
                await websocket.send_json({
                    "type": "live_prediction",
                    "predicted_label": "Model Unavailable",
                    "confidence": 0.0,
                    "meets_threshold": False,
                    "top_k": [],
                    "state": state,
                    "state_code": "STATE_G_MODEL_UNAVAILABLE",
                    "is_provisional": True,
                })
                continue

            inference_ms = 0.0

            # Compute instant wrist motion speed
            instant_speed = 0.0
            if len(wrist_history) >= 2:
                instant_speed = float(
                    np.hypot(
                        wrist_history[-1][0] - wrist_history[-2][0],
                        wrist_history[-1][1] - wrist_history[-2][1],
                    )
                )
                speed_history.append(instant_speed)

            avg_speed = float(np.mean(speed_history)) if speed_history else 0.0
            is_moving = avg_speed >= 0.007

            # 2. Dynamic motion stroke inference
            if hands_present:
                buf_len = len(sequence_buffer)

                # If buffer is still accumulating initial stroke frames
                if buf_len < min_sequence_frames:
                    status_text = "Capturing sign motion..." if is_moving else "Ready: Perform sign movement"
                    await websocket.send_json({
                        "type": "live_prediction",
                        "predicted_label": status_text,
                        "confidence": 0.0,
                        "meets_threshold": False,
                        "top_k": [],
                        "state": state,
                        "state_code": "STATE_C_COLLECTING",
                        "is_provisional": True,
                        "latency_ms": round(extraction_ms, 1),
                    })
                    continue

                # If hand is held completely stationary, suppress zero-velocity biased predictions
                if not is_moving and current_candidate is None:
                    await websocket.send_json({
                        "type": "live_prediction",
                        "predicted_label": "Stationary hand (Move to sign)",
                        "confidence": 0.0,
                        "meets_threshold": False,
                        "top_k": [],
                        "state": state,
                        "state_code": "STATE_C_COLLECTING",
                        "is_provisional": True,
                        "latency_ms": round(extraction_ms, 1),
                    })
                    continue

                # Evaluate actual dynamic motion trajectory with prior debiasing and masks
                seq_array = np.stack(list(sequence_buffer))
                mask_array = np.stack(list(mask_buffer))
                t1 = time.perf_counter()
                try:
                    pred_res = predictor.predict_label(
                        seq_array,
                        mask=mask_array,
                        confidence_threshold=confidence_threshold,
                        calibrate=True,
                        calibration_alpha=0.85,
                    )
                except Exception as exc:
                    logger.debug("Prediction failed: %s", exc)
                    continue
                inference_ms = (time.perf_counter() - t1) * 1000.0

                pred_label = pred_res["predicted_label"]
                conf = pred_res["confidence"]
                meets_threshold = pred_res["meets_threshold"]
                top_k = pred_res.get("top_k", [])

                # Emit live prediction
                await websocket.send_json({
                    "type": "live_prediction",
                    "predicted_label": pred_label,
                    "confidence": conf,
                    "meets_threshold": meets_threshold,
                    "top_k": top_k,
                    "state": state,
                    "state_code": state_code,
                    "is_provisional": True,
                    "latency_ms": round(extraction_ms + inference_ms, 1),
                })

                # Mode A: Guided Word Accumulation
                if mode == "guided":
                    if meets_threshold and pred_label not in {"Unknown (Low Confidence)", "Model Unavailable"}:
                        candidate_history.append(pred_label)
                        most_common, count = collections.Counter(candidate_history).most_common(1)[0]
                        current_candidate = most_common
                        current_confidence = conf
                        progress = min(1.0, count / debounce_frames)

                        await websocket.send_json({
                            "type": "word_candidate",
                            "word": current_candidate,
                            "confidence": current_confidence,
                            "stability_count": count,
                            "target_count": debounce_frames,
                            "progress": progress,
                        })

                        # Commit confirmed word if stability threshold is satisfied
                        if count >= debounce_frames and state != "COMMITTED":
                            is_new_word = current_candidate != last_committed_word
                            is_new_gesture = (now - last_hand_activity_time) < 0.2

                            if is_new_word or is_new_gesture:
                                token_id = str(uuid.uuid4())[:8]
                                await websocket.send_json({
                                    "type": "word_committed",
                                    "token_id": token_id,
                                    "word": current_candidate,
                                    "confidence": current_confidence,
                                    "timestamp": time.time(),
                                    "mode": "guided",
                                })
                                last_committed_word = current_candidate
                                state = "COMMITTED"
                                sequence_buffer.clear()
                                candidate_history.clear()
                                speed_history.clear()
                    else:
                        if state != "COMMITTED":
                            await websocket.send_json({
                                "type": "word_candidate",
                                "word": None,
                                "confidence": conf,
                                "stability_count": 0,
                                "target_count": debounce_frames,
                                "progress": 0.0,
                            })

                # Mode B: Continuous Sign Recognition (Experimental)
                elif mode == "continuous":
                    if meets_threshold and pred_label not in {"Unknown (Low Confidence)", "Model Unavailable"}:
                        await websocket.send_json({
                            "type": "word_candidate",
                            "word": pred_label,
                            "confidence": conf,
                            "mode": "continuous_experimental",
                            "progress": 1.0,
                        })

            # Send periodic performance metrics (every 5 frames)
            if frame_counter % 5 == 0:
                elapsed = max(1e-3, now - start_time)
                fps = frame_counter / elapsed
                await websocket.send_json({
                    "type": "performance",
                    "fps": round(fps, 1),
                    "extraction_ms": round(extraction_ms, 1),
                    "inference_ms": round(inference_ms, 1),
                    "total_latency_ms": round(extraction_ms + inference_ms, 1),
                    "frame_count": frame_counter,
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected cleanly. Session ID: %s", session_id)
    except Exception as exc:
        if "disconnect message has been received" in str(exc) or "WebSocket is not connected" in str(exc):
            logger.info("WebSocket client disconnected cleanly. Session ID: %s", session_id)
        else:
            logger.warning("Unexpected error in WebSocket loop: %s", exc)


# Mount static production build of frontend if present
_dist_dir = Path("app/frontend/dist")
if _dist_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist_dir), html=True), name="frontend")
