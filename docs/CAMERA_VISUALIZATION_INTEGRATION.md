# SignFlow Live Camera Visualization & Upper-Body Framing Integration

**Date:** 30 October 2026  
**Module:** Live Camera Workspace (`CameraPanel.tsx`), WebSocket API (`/ws/live`), Multi-Modality Landmarks  
**Status:** Implemented, Tested, and Verified  

---

## 1. Overview & Architectural Principle

SignFlow integrates a high-performance live camera visualization and upper-body tracking experience inspired by modern computer-vision interfaces (such as the Hugging Face Space `naik123/sign-language-reader`), while strictly preserving SignFlow's end-to-end Python MediaPipe Holistic extraction and trained WLASL-100 word-level recognition models.

### Key Principles
1. **Frontend Visual Reference Only:** Ported clean aesthetic styling (smooth bone linkages, circular joint markers, fingertip highlights, multi-layer toggles, and truthful FPS/latency indicators).
2. **Preserve Word-Level Recognizer:** Retained SignFlow's 3-Way Probability Ensemble (`weights = [0.25, 0.10, 0.65]`), 553-point holistic pipeline, and WLASL-100 vocabulary. Did not import static alphabet heuristic rules or client-side single-hand recognizers.
3. **Single Backend Extraction Source:** The backend MediaPipe Holistic task asset extracts 553 points per frame (`left_hand: 21`, `right_hand: 21`, `pose: 33`, `face: 478`). The existing `/ws/live` WebSocket message was extended backward-compatibly to transport `pose_landmarks` and `face_landmarks` coordinates and validities. No redundant browser-side detector is run.

---

## 2. WebSocket Protocol Extension (`/ws/live`)

### Outgoing `hand_landmarks` Payload Contract
The WebSocket message preserves all existing fields and adds structured pose and face landmark arrays:

```json
{
  "type": "hand_landmarks",
  "frame_id": 104,
  "timestamp_ms": 1730315200123,
  "left_landmarks": [
    { "x": 0.3541, "y": 0.6124, "z": -0.0215, "valid": true },
    "... 21 points total ..."
  ],
  "right_landmarks": [
    { "x": 0.6214, "y": 0.5891, "z": -0.0184, "valid": true },
    "... 21 points total ..."
  ],
  "pose_landmarks": [
    { "x": 0.4952, "y": 0.3120, "z": -0.1042, "valid": true },
    "... 33 points total (indices 0..32) ..."
  ],
  "face_landmarks": [
    { "x": 0.5012, "y": 0.2841, "z": -0.0521, "valid": true },
    "... 478 points total (indices 0..477) ..."
  ],
  "left_box": { "xmin": 0.30, "ymin": 0.52, "xmax": 0.41, "ymax": 0.70, "width": 0.11, "height": 0.18 },
  "right_box": { "xmin": 0.57, "ymin": 0.50, "xmax": 0.68, "ymax": 0.68, "width": 0.11, "height": 0.18 },
  "hands_detected": true,
  "left_hand": true,
  "right_hand": true,
  "pose": true,
  "face": true,
  "valid_points": 542,
  "quality": {
    "score": 95,
    "buffer_completeness": 1.0
  },
  "state_code": "STATE_C_COLLECTING"
}
```

### Point Index & Modality Offsets
Slices are aligned with `src/sign_language/landmarks/schema.py`:
- `left_hand`: indices `0..20` (21 points)
- `right_hand`: indices `21..41` (21 points)
- `pose`: indices `42..74` (33 points)
- `face`: indices `75..552` (478 points)
- Floating point values are formatted with `round(val, 4)` for efficient payload compression without sacrificing sub-pixel precision.

---

## 3. Overlay Topology & Drawing Rules

### 1. Hands (21 points each)
- **Left Hand:** Indigo (`#6366f1`), glow `#4f46e5`.
- **Right Hand:** Teal (`#14b8a6`), glow `#0d9488`.
- **Connections:** Thumb (0-4), Index (0-8), Middle (0-12), Ring (0-16), Pinky (0-20), and Knuckle bridges (5-9-13-17).
- **Joints:** Wrist radius 5.5px, normal joints 3.5px, fingertips (indices 4, 8, 12, 16, 20) highlighted in white (`#ffffff`) with 4.5px radius and color ring.
- **Both hands can be drawn simultaneously.**

### 2. Upper-Body Pose (33 points)
- **Color:** Sky Blue (`#38bdf8`), glow `#0284c7`.
- **Connections:**
  - Shoulder line: `[11, 12]`
  - Torso anchors: `[11, 23]`, `[12, 24]`, `[23, 24]`
  - Left arm chain: `[11, 13]`, `[13, 15]`
  - Right arm chain: `[12, 14]`, `[14, 16]`
  - Neck / head anchors: `[11, 0]`, `[12, 0]`
  - Facial anchors on pose: `[0, 1]`, `[1, 2]`, `[2, 3]`, `[3, 7]` (left eye/ear), `[0, 4]`, `[4, 5]`, `[5, 6]`, `[6, 8]` (right eye/ear), `[9, 10]` (mouth).
- **Rule:** A bone line is stroked **only when both endpoints are valid (`p1.valid && p2.valid`)**. Invalid joints never produce spurious lines.

### 3. Face Contours (Curated Subset of 478-Point Topology)
- **Color:** Rose Pink (`#ec4899`), glow `#be185d`.
- **Curated Paths:**
  - **Face Oval:** `10 -> 338 -> 297 -> 332 -> 284 -> 251 -> 389 -> 356 -> 454 -> 323 -> 361 -> 288 -> 397 -> 365 -> 379 -> 378 -> 400 -> 377 -> 152 -> 148 -> 176 -> 149 -> 150 -> 136 -> 172 -> 58 -> 132 -> 93 -> 234 -> 127 -> 162 -> 21 -> 54 -> 103 -> 67 -> 109 -> 10`
  - **Right Eyebrow:** `70 -> 63 -> 105 -> 66 -> 107`
  - **Left Eyebrow:** `336 -> 296 -> 334 -> 293 -> 300`
  - **Right Eye:** `33 -> 7 -> 163 -> 144 -> 145 -> 153 -> 154 -> 155 -> 133 -> 173 -> 157 -> 158 -> 159 -> 160 -> 161 -> 246 -> 33`
  - **Left Eye:** `362 -> 382 -> 381 -> 380 -> 374 -> 373 -> 390 -> 249 -> 263 -> 466 -> 388 -> 387 -> 386 -> 385 -> 384 -> 398 -> 362`
  - **Lips Outer:** `61 -> 146 -> 91 -> 181 -> 84 -> 17 -> 314 -> 405 -> 321 -> 375 -> 291 -> 409 -> 270 -> 269 -> 267 -> 0 -> 37 -> 39 -> 40 -> 185 -> 61`
  - **Lips Inner:** `78 -> 95 -> 88 -> 178 -> 87 -> 14 -> 317 -> 402 -> 318 -> 324 -> 308 -> 415 -> 310 -> 311 -> 312 -> 13 -> 82 -> 81 -> 80 -> 191 -> 78`
  - **Nose Bridge & Tip:** `168 -> 6 -> 197 -> 195 -> 5 -> 4 -> 1` and `98 -> 97 -> 2 -> 326 -> 327`
- Avoids dense triangular meshes to ensure high legibility and fast canvas rendering.

---

## 4. Upper-Body Framing & Camera Behavior

### 1. Default View: `upper_body`
- Displays the **full source camera frame uncropped** using `object-contain`.
- Overlay canvas mirrors the identical intrinsic aspect ratio and CSS letterbox bounding box as `<video>`, maintaining 1:1 pixel alignment across all webcam aspect ratios (`4:3`, `16:9`, portrait).

### 2. Coordinate Mapping & Mirroring
- **Source coordinate mapping:**
  - `mapX(x) = (mirrorCamera ? (1.0 - x) : x) * canvasWidth`
  - `mapY(y) = y * canvasHeight`
- Presentation mirroring is applied strictly once:
  - `<video>` is mirrored via CSS (`-scale-x-100`).
  - Overlay `<canvas>` coordinates are mapped with `mapX(x)` and is NOT mirrored in CSS.
  - Hidden capture canvas and transmitted JPEG remain in canonical unmirrored camera coordinates as expected by the inference engine.

### 3. Dynamic Framing Guidance
The guide continuously evaluates landmark status to provide truthful positioning feedback:
- Missing pose: *"Torso/shoulders not detected — position upper body in frame"*
- Missing face: *"Face not detected — position head in upper center frame"*
- Hands near border (`< 0.05` or `> 0.95`): *"Hands near frame edge — step back slightly (~1.5m) for full reach"*
- Single hand: *"{Left|Right} hand active — bring second hand in frame if signing two-handed"*
- Both hands active: *"Both hands and upper-body tracking — framing optimal"*

### 4. Hardware Zoom Inspection
- When camera starts, inspects `videoTrack.getCapabilities()`.
- If optical zoom is supported, exposes a physical zoom slider using `applyConstraints({ advanced: [{ zoom: val }] })`.
- If unsupported, falls back gracefully without forcing constraints or applying deceptive CSS scaling, advising the signer to step back.

### 5. Dual-Hand Detail View
- In `focus` and `split` modes, computes the smoothed **union bounding box** of both hands with 4% padding margin so neither hand is discarded.

---

## 5. Verification & Test Suite

### Frontend Tests (`npm test`)
- `src/components/CameraPanel.test.tsx` (6 tests):
  - Upper Body view mode and controls rendering.
  - Simultaneous dual-hand rendering with distinct colors.
  - Pose and face status indicators.
  - Layer toggles (Hands, Pose, Face, Points).
  - View mode switching (Upper Body, Full, Hands Detail, Split).
  - Stream stopping and animation frame cleanup on unmount.
- Total Frontend Suite: **22 passed** (100% pass rate).

### Backend Tests (`pytest tests/test_api.py`)
- Shape verification: `len(pose_landmarks) == 33` and `len(face_landmarks) == 478`.
- Schema verification: each landmark point has `x: float`, `y: float`, `z: float`, `valid: bool`.
- Point mask verification: empty/black frames yield `valid: False` with finite numerical coordinates.
- Backward compatibility: `hands_detected`, `left_hand`, `right_hand`, `pose`, `face`, `frame_id`.
- Total Backend Suite: **19 passed** (100% pass rate).

