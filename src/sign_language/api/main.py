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

    # Initialize RealTimePredictor
    try:
        if model_path and Path(model_path).is_file():
            _predictor = RealTimePredictor(model_path=model_path, config_path=config_path)
            logger.info("RealTimePredictor successfully initialized with model %s", model_path)
        else:
            _predictor = RealTimePredictor(config_path=config_path)
            logger.info("RealTimePredictor initialized from config %s (untrained baseline)", config_path)
        _init_error = None
    except Exception as exc:
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
        try:
            _predictor = RealTimePredictor(config_path="configs/base.yaml")
            _init_error = None
        except Exception as exc:
            _init_error = str(exc)
            raise HTTPException(status_code=503, detail=f"Predictor service unavailable: {_init_error}")
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

    # Connection-level state
    mode = "guided"  # "guided" or "continuous"
    confidence_threshold = 0.40
    debounce_frames = 5
    pause_threshold_sec = 0.6
    min_sequence_frames = min(16, seq_len)

    sequence_buffer: collections.deque = collections.deque(maxlen=seq_len)
    candidate_history: collections.deque = collections.deque(maxlen=debounce_frames)

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
                    candidate_history.clear()
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

            # 1. Landmark extraction
            t0 = time.perf_counter()
            try:
                structured = extractor.extract_structured_frame(frame, timestamp_ms=int(ts_ms))
                obs = extractor.extract_observation(frame, timestamp_ms=int(ts_ms))
            except Exception as exc:
                logger.debug("Extraction failed for frame: %s", exc)
                continue
            extraction_ms = (time.perf_counter() - t0) * 1000.0

            # Compute landmark presence
            left_hand = bool(structured.point_mask[:21].sum() > 0)
            right_hand = bool(structured.point_mask[21:42].sum() > 0)
            pose = bool(structured.point_mask[42:75].sum() > 0)
            face = bool(structured.point_mask[75:].sum() > 0)
            hands_present = left_hand or right_hand
            valid_points = int(structured.point_mask.sum())

            await websocket.send_json({
                "type": "landmarks_status",
                "hands_detected": hands_present,
                "left_hand": left_hand,
                "right_hand": right_hand,
                "pose": pose,
                "face": face,
                "valid_points": valid_points,
            })

            # Hand activity tracking for neutral pause detection
            if hands_present:
                last_hand_activity_time = now
                if state == "WAITING":
                    state = "CAPTURING"
            else:
                if (now - last_hand_activity_time) > pause_threshold_sec:
                    # Neutral pause reached
                    if state == "COMMITTED":
                        state = "WAITING"
                        last_committed_word = None  # Allow repeating the same word on next gesture
                    elif state == "CAPTURING" and len(sequence_buffer) < min_sequence_frames:
                        state = "WAITING"

            # Append to rolling sequence buffer
            sequence_buffer.append(obs.landmarks)

            # Ensure model is ready
            if predictor is None:
                await websocket.send_json({
                    "type": "live_prediction",
                    "predicted_label": "Model Unavailable",
                    "confidence": 0.0,
                    "meets_threshold": False,
                    "top_k": [],
                    "state": state,
                    "is_provisional": True,
                })
                continue

            inference_ms = 0.0

            # 2. Model inference when minimum sequence length is reached
            if len(sequence_buffer) >= min_sequence_frames:
                seq_array = np.stack(list(sequence_buffer))
                t1 = time.perf_counter()
                try:
                    pred_res = predictor.predict_label(
                        seq_array, confidence_threshold=confidence_threshold
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
