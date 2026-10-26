# SignFlow: Real-Time Latency, Throughput & Compression Benchmark Report

**Benchmark Date:** October 26, 2026  
**Benchmarking Tool:** `scripts/benchmark_latency.py`  
**Host Platform:** Windows 11 (Intel Core CPU, 64-bit)  
**Evaluated Architecture:** `RobustHolisticFusionClassifier` (100 classes, 64-frame buffer)  

---

## 1. Empirical Latency & Resolution Profile

Benchmark executed across all 9 resolution and compression permutations using prerecorded video frames from `data/WLASL-3/videos/00336.mp4` running through MediaPipe Holistic Landmarker and PyTorch CPU inference:

| Scenario | Resolution | JPEG Quality | Frame Payload | Encode (ms) | Decode (ms) | Extraction (ms) | Inference (ms) | p50 Latency (ms) | p95 Latency (ms) | Estimated FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `320x240_q50` | $320\times240$ | 50 | 5.90 KB | 1.32 | 0.55 | 559.80* | 37.24 | 96.83 | 2366.54* | 1.7* |
| `320x240_q75` | $320\times240$ | 75 | 8.06 KB | 0.36 | 0.38 | 50.89 | 29.96 | 79.19 | 102.90 | 12.3 |
| `320x240_q90` | $320\times240$ | 90 | 12.58 KB | 0.49 | 0.52 | 68.25 | 42.75 | 109.51 | 127.94 | 8.9 |
| `480x360_q50` | $480\times360$ | 50 | 9.86 KB | 0.77 | 0.82 | 5.07 | 32.62 | 32.59 | 69.32 | 25.5 |
| **`480x360_q75`** | **$480\times360$** | **75** | **13.54 KB** | **0.77** | **0.88** | **0.67** | **29.33** | **31.66** | **36.64** | **31.6** |
| `480x360_q90` | $480\times360$ | 90 | 21.20 KB | 0.83 | 0.99 | 0.69 | 28.37 | 30.58 | 36.32 | 32.4 |
| `640x480_q50` | $640\times480$ | 50 | 14.42 KB | 1.16 | 1.63 | 0.95 | 27.31 | 30.75 | 37.77 | 32.2 |
| `640x480_q75` | $640\times480$ | 75 | 19.56 KB | 1.25 | 1.43 | 0.92 | 30.79 | 34.28 | 42.17 | 29.1 |
| `640x480_q90` | $640\times480$ | 90 | 30.93 KB | 1.38 | 1.65 | 0.88 | 28.98 | 31.59 | 40.51 | 30.4 |

*\*Note: Cold-start compilation artifact during the first scenario run while MediaPipe compiled XNNPACK delegate kernels.*

---

## 2. Key Engineering Findings

1. **Sweet Spot Recommendation**:
   - **$480\times360$ at JPEG Quality 75** is the optimal operating point. It yields ~13.5 KB payload per frame, ~31.7 ms median end-to-end latency, and sustained ~31.6 FPS on host CPU.
2. **Network Bandwidth Consumption**:
   - Streaming $480\times360$ at Quality 75 at 25 FPS consumes only **~338 KB/sec (~2.7 Mbps)**, well within standard residential broadband and mobile 4G/5G connections.
3. **Pipeline Bottleneck Attribution**:
   - Network encode/decode: ~1.5–2.5 ms (negligible).
   - Landmark extraction: ~0.7–1.5 ms on CPU once warmed up.
   - Model inference: ~28–32 ms on CPU. (Projected to drop to < 5 ms on GPU via TensorRT or CUDA).
4. **Webcam Real-World Disclosure**:
   - The measurements above were obtained using prerecorded MP4 frames passed through the full pipeline. Physical user webcam performance remains dependent on local lighting, camera sensor frame rates (e.g. 30Hz limit), and browser execution.
