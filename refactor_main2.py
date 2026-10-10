import sys

with open("src/sign_language/api/main.py", "r") as f:
    code = f.read()

# Define the new SessionState class
state_class = """
class SessionState:
    def __init__(self, seq_len: int):
        import collections, time
        self.seq_len = seq_len
        self.mode = "guided"
        self.confidence_threshold = 0.28
        self.debounce_frames = 2
        self.pause_threshold_sec = 0.5
        self.min_sequence_frames = 10

        self.sequence_buffer: collections.deque = collections.deque(maxlen=self.seq_len)
        self.mask_buffer: collections.deque = collections.deque(maxlen=self.seq_len)
        self.candidate_history: collections.deque = collections.deque(maxlen=self.debounce_frames)
        self.recent_replay_frames: collections.deque = collections.deque(maxlen=64)
        self.wrist_history: collections.deque = collections.deque(maxlen=5)
        self.speed_history: collections.deque = collections.deque(maxlen=10)

        self.last_brightness = None
        self.last_blur_score = None

        self.current_candidate = None
        self.current_confidence = 0.0
        self.last_committed_word = None
        
        # continuous mode specific
        self.continuous_candidate_history = collections.deque(maxlen=5) # longer hysteresis for continuous
        self.continuous_cooldown_frames = 0
        self.continuous_last_emitted = None

        self.state = "WAITING"
        self.last_hand_activity_time = time.time()
        self.frame_counter = 0
        self.start_time = time.time()

    def update_config(self, payload: dict):
        import collections
        if "mode" in payload and payload["mode"] in {"guided", "continuous"}:
            self.mode = payload["mode"]
        if "confidence_threshold" in payload:
            self.confidence_threshold = float(payload["confidence_threshold"])
        if "debounce_frames" in payload:
            self.debounce_frames = max(2, int(payload["debounce_frames"]))
            self.candidate_history = collections.deque(
                list(self.candidate_history), maxlen=self.debounce_frames
            )
        if "pause_threshold_sec" in payload:
            self.pause_threshold_sec = float(payload["pause_threshold_sec"])

    def full_reset(self):
        self.sequence_buffer.clear()
        self.mask_buffer.clear()
        self.candidate_history.clear()
        self.wrist_history.clear()
        self.speed_history.clear()
        self.continuous_candidate_history.clear()
        self.continuous_cooldown_frames = 0
        self.continuous_last_emitted = None
        self.current_candidate = None
        self.current_confidence = 0.0
        self.last_committed_word = None
        self.state = "WAITING"

    def transition_to_committed(self, word: str):
        self.last_committed_word = word
        self.state = "COMMITTED"
        self.current_candidate = None
        self.sequence_buffer.clear()
        self.mask_buffer.clear()
        self.candidate_history.clear()
        self.speed_history.clear()
        self.wrist_history.clear()

    def handle_neutral_pause(self):
        if self.state in {"COMMITTED", "CAPTURING", "CANDIDATE"}:
            self.state = "WAITING"
            self.sequence_buffer.clear()
            self.mask_buffer.clear()
            self.wrist_history.clear()
            self.speed_history.clear()
            self.continuous_last_emitted = None # Allow repeating words after pause
            
    def append_frame(self, obs_landmarks, obs_mask):
        self.sequence_buffer.append(obs_landmarks)
        self.mask_buffer.append(obs_mask)

"""

# Extract everything up to websocket_live_recognition
ws_start = code.find('@app.websocket("/ws/live")')
if ws_start == -1:
    print("Could not find websocket endpoint")
    sys.exit(1)

code_before = code[:ws_start]

# We will completely replace the websocket_live_recognition function
new_ws_func = """
@app.websocket("/ws/live")
async def websocket_live_recognition(websocket: WebSocket) -> None:
    \"\"\"Persistent bidirectional WebSocket connection for real-time ASL recognition.\"\"\"
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info("WebSocket client connected. Session ID: %s", session_id)

    predictor = _predictor
    extractor = _extractor

    model_ready = predictor is not None and getattr(predictor, "model", None) is not None
    extractor_ready = extractor is not None

    seq_len = predictor.sequence_length if predictor else 64
    raw_dim = (
        predictor.input_dim // 3
        if (predictor and predictor.input_dim >= 4977)
        else (predictor.input_dim if predictor else 1659)
    )

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

    s = SessionState(seq_len)

    try:
        while True:
            message = await websocket.receive()
            now = time.time()
            if "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON format."})
                    continue
                msg_type = payload.get("type", "")

                if msg_type == "request_replay":
                    await websocket.send_json({
                        "type": "replay_buffer",
                        "frames": list(s.recent_replay_frames),
                    })
                    continue
                elif msg_type == "config":
                    s.update_config(payload)
                    await websocket.send_json({
                        "type": "config_updated",
                        "mode": s.mode,
                        "confidence_threshold": s.confidence_threshold,
                        "debounce_frames": s.debounce_frames,
                        "pause_threshold_sec": s.pause_threshold_sec,
                    })
                    continue
                elif msg_type == "confirm_word":
                    if s.current_candidate:
                        token_id = str(uuid.uuid4())[:8]
                        await websocket.send_json({
                            "type": "word_committed",
                            "token_id": token_id,
                            "word": s.current_candidate,
                            "confidence": s.current_confidence,
                            "timestamp": time.time(),
                            "mode": s.mode,
                            "source": "manual_confirm",
                        })
                        s.transition_to_committed(s.current_candidate)
                    continue
                elif msg_type == "reject_candidate":
                    s.current_candidate = None
                    s.current_confidence = 0.0
                    s.candidate_history.clear()
                    s.state = "WAITING"
                    await websocket.send_json({
                        "type": "word_candidate",
                        "word": None,
                        "confidence": 0.0,
                        "stability_count": 0,
                        "target_count": s.debounce_frames,
                        "progress": 0.0,
                    })
                    continue
                elif msg_type == "reset":
                    s.full_reset()
                    await websocket.send_json({"type": "reset_ack"})
                    continue
                elif msg_type == "frame":
                    frame_data = payload.get("data", "")
                    ts_ms = payload.get("timestamp_ms", int(time.time() * 1000))
                    if not frame_data: continue
                    if "," in frame_data:
                        frame_data = frame_data.split(",", 1)[1]
                    try:
                        img_bytes = base64.b64decode(frame_data)
                        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    except Exception as exc:
                        await websocket.send_json({"type": "warning", "message": f"Frame decode error: {exc}"})
                        continue
                    if frame is None: continue
                else: continue
            elif "bytes" in message and message["bytes"]:
                ts_ms = int(time.time() * 1000)
                try:
                    img_arr = np.frombuffer(message["bytes"], dtype=np.uint8)
                    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                except Exception as exc:
                    await websocket.send_json({"type": "warning", "message": f"Binary decode error: {exc}"})
                    continue
                if frame is None: continue
            else: continue

            s.frame_counter += 1
            if extractor is None:
                await websocket.send_json({
                    "type": "landmarks_status",
                    "hands_detected": False,
                    "left_hand": False,
                    "right_hand": False,
                    "pose": False,
                    "face": False,
                    "valid_points": 0,
                    "error": "MediaPipe unavailable.",
                })
                continue

            t0 = time.perf_counter()
            try:
                structured = extractor.extract_structured_frame(frame, timestamp_ms=int(ts_ms))
            except Exception: continue
            extraction_ms = (time.perf_counter() - t0) * 1000.0

            selected_modalities = ["left_hand", "right_hand"]
            if extractor.representation in {"hands_pose", "holistic"}: selected_modalities.append("pose")
            if extractor.representation == "holistic": selected_modalities.append("face")

            mod_values, mod_masks = [], []
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

            coords, mask = structured.coordinates, structured.point_mask
            left_hand = bool(mask[:21].sum() > 0)
            right_hand = bool(mask[21:42].sum() > 0)
            pose = bool(mask[42:75].sum() > 0)
            face = bool(mask[75:].sum() > 0)
            hands_present = left_hand or right_hand
            valid_points = int(mask.sum())
            num_hands = (1 if left_hand else 0) + (1 if right_hand else 0)

            left_pts = [{"x": float(coords[i, 0]), "y": float(coords[i, 1]), "z": float(coords[i, 2]), "valid": bool(mask[i] > 0)} for i in range(21)]
            right_pts = [{"x": float(coords[i, 0]), "y": float(coords[i, 1]), "z": float(coords[i, 2]), "valid": bool(mask[i] > 0)} for i in range(21, 42)]
            
            # Simple Quality Coach
            coach_score = 100
            if not hands_present:
                coach_score -= 60
            buf_completeness = min(1.0, len(s.sequence_buffer) / max(1, s.min_sequence_frames))

            state_code = "STATE_F_UNCERTAIN"
            if predictor is None: state_code = "STATE_G_MODEL_UNAVAILABLE"
            elif not hands_present: state_code = "STATE_B_NO_HAND"
            elif len(s.sequence_buffer) < s.min_sequence_frames: state_code = "STATE_C_COLLECTING"
            elif s.state == "COMMITTED": state_code = "STATE_E_COMMITTED"
            elif s.current_candidate and s.current_candidate not in {"Unknown (Low Confidence)", "Model Unavailable"}: state_code = "STATE_D_CANDIDATE"

            await websocket.send_json({
                "type": "hand_landmarks",
                "left_landmarks": left_pts,
                "right_landmarks": right_pts,
                "left_box": None, "right_box": None,
                "hands_detected": hands_present,
                "left_hand": left_hand, "right_hand": right_hand,
                "pose": pose, "face": face,
                "valid_points": valid_points,
                "quality": {"score": max(5, coach_score), "buffer_completeness": buf_completeness},
                "state_code": state_code,
                "timestamp_ms": ts_ms,
            })
            
            s.recent_replay_frames.append({
                "frame_idx": s.frame_counter, "timestamp_ms": ts_ms,
                "left_landmarks": left_pts, "right_landmarks": right_pts,
                "hands_detected": hands_present, "valid_points": valid_points,
                "state_code": state_code, "candidate": s.current_candidate,
                "confidence": s.current_confidence, "features": obs_landmarks.tolist(),
            })

            # Hand activity tracking
            if hands_present:
                s.last_hand_activity_time = now
                if s.state == "WAITING": s.state = "CAPTURING"
                s.append_frame(obs_landmarks, obs_mask)
            else:
                if (now - s.last_hand_activity_time) > s.pause_threshold_sec:
                    s.handle_neutral_pause()

            if predictor is None: continue
            
            inference_ms = 0.0
            
            # Speed logic
            if hands_present:
                wrist_pos = None
                if right_hand and mask[21] > 0: wrist_pos = (coords[21,0], coords[21,1])
                elif left_hand and mask[0] > 0: wrist_pos = (coords[0,0], coords[0,1])
                if wrist_pos:
                    s.wrist_history.append(wrist_pos)
                    if len(s.wrist_history) >= 2:
                        s.speed_history.append(float(np.hypot(s.wrist_history[-1][0]-s.wrist_history[-2][0], s.wrist_history[-1][1]-s.wrist_history[-2][1])))

            avg_speed = float(np.mean(s.speed_history)) if s.speed_history else 0.0
            is_moving = avg_speed >= 0.007

            if hands_present:
                if len(s.sequence_buffer) < s.min_sequence_frames: continue
                if not is_moving and s.current_candidate is None: continue

                # Inference
                seq_array = np.stack(list(s.sequence_buffer))
                mask_array = np.stack(list(s.mask_buffer))
                
                # Cooldown logic for continuous mode
                if s.mode == "continuous" and s.continuous_cooldown_frames > 0:
                    s.continuous_cooldown_frames -= 1
                    continue
                    
                t1 = time.perf_counter()
                try:
                    pred_res = predictor.predict_label(
                        seq_array, mask=mask_array, confidence_threshold=s.confidence_threshold, calibrate=True, calibration_alpha=0.85
                    )
                except Exception as exc: continue
                inference_ms = (time.perf_counter() - t1) * 1000.0

                pred_label = pred_res["predicted_label"]
                conf = pred_res["confidence"]
                meets_threshold = pred_res["meets_threshold"]

                await websocket.send_json({
                    "type": "live_prediction",
                    "predicted_label": pred_label, "confidence": conf,
                    "meets_threshold": meets_threshold, "top_k": pred_res.get("top_k", []),
                    "state": s.state, "state_code": state_code,
                    "is_provisional": True, "latency_ms": round(extraction_ms + inference_ms, 1),
                })

                if s.mode == "guided":
                    if meets_threshold and pred_label not in {"Unknown (Low Confidence)", "Model Unavailable"}:
                        s.candidate_history.append(pred_label)
                        most_common, count = collections.Counter(s.candidate_history).most_common(1)[0]
                        s.current_candidate = most_common
                        s.current_confidence = conf
                        progress = min(1.0, count / s.debounce_frames)
                        await websocket.send_json({
                            "type": "word_candidate", "word": s.current_candidate, "confidence": s.current_confidence,
                            "stability_count": count, "target_count": s.debounce_frames, "progress": progress,
                        })
                        if count >= s.debounce_frames and s.state != "COMMITTED":
                            is_new_word = s.current_candidate != s.last_committed_word
                            is_new_gesture = (now - s.last_hand_activity_time) < 0.2
                            if is_new_word or is_new_gesture:
                                await websocket.send_json({
                                    "type": "word_committed", "token_id": str(uuid.uuid4())[:8],
                                    "word": s.current_candidate, "confidence": s.current_confidence,
                                    "timestamp": time.time(), "mode": "guided",
                                })
                                s.transition_to_committed(s.current_candidate)
                    else:
                        if s.state != "COMMITTED":
                            await websocket.send_json({
                                "type": "word_candidate", "word": None, "confidence": conf,
                                "stability_count": 0, "target_count": s.debounce_frames, "progress": 0.0,
                            })
                elif s.mode == "continuous":
                    if meets_threshold and pred_label not in {"Unknown (Low Confidence)", "Model Unavailable"}:
                        s.continuous_candidate_history.append(pred_label)
                        if len(s.continuous_candidate_history) == s.continuous_candidate_history.maxlen:
                            # Requires the same label to be predicted consistently across the history buffer
                            most_common, count = collections.Counter(s.continuous_candidate_history).most_common(1)[0]
                            # Confirm if it's stable and different from the last emitted word (or a new gesture phase)
                            if count >= 4 and most_common != s.continuous_last_emitted:
                                await websocket.send_json({
                                    "type": "word_committed", "token_id": str(uuid.uuid4())[:8],
                                    "word": most_common, "confidence": conf,
                                    "timestamp": time.time(), "mode": "continuous", "source": "auto_continuous"
                                })
                                s.continuous_last_emitted = most_common
                                s.continuous_cooldown_frames = 15 # Wait a few frames before recognizing the next sign
                                s.continuous_candidate_history.clear()
                                # Clear rolling sequence buffers to prevent stale frames leaking into the next sign prediction
                                s.sequence_buffer.clear()
                                s.mask_buffer.clear()
                            
            if s.frame_counter % 5 == 0:
                elapsed = max(1e-3, now - s.start_time)
                await websocket.send_json({
                    "type": "performance",
                    "fps": round(s.frame_counter / elapsed, 1),
                    "extraction_ms": round(extraction_ms, 1),
                    "inference_ms": round(inference_ms, 1),
                    "total_latency_ms": round(extraction_ms + inference_ms, 1),
                    "frame_count": s.frame_counter,
                })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected cleanly. Session ID: %s", session_id)
    except Exception as exc:
        logger.warning("Unexpected error in WebSocket loop: %s", exc)

# Mount static production build of frontend if present
_dist_dir = Path("app/frontend/dist")
if _dist_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist_dir), html=True), name="frontend")
"""

with open("src/sign_language/api/main.py", "w") as f:
    f.write(code_before + state_class + new_ws_func)

print("Replaced main.py successfully")
