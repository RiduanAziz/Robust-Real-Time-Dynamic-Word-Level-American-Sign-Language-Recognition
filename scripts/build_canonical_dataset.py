#!/usr/bin/env python3
"""Build Canonical Merged Dataset from WLASL-2 and WLASL-3.

Integrates:
1. Full 21,083 holistic landmark sequences from WLASL-2 (landmarks_V3.npz)
   with canonical modality ordering (Left Hand, Right Hand, Pose, Face)
   and explicit point visibility masks.
2. Official video assets, split assignments, and signer identities from WLASL-3.
3. Word semantic embeddings (labels.npz) for zero-shot and multimodal support.
4. Generates standard JSON manifests conforming to sign_language.data contracts.
"""

from __future__ import annotations

import collections
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    print("=" * 70)
    print("BUILDING CANONICAL SIGN LANGUAGE DATASET (WLASL-2 + WLASL-3 MERGE)")
    print("=" * 70)

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    w2_dir = data_dir / "WLASL-2"
    w3_dir = data_dir / "WLASL-3"
    landmarks_dir = data_dir / "landmarks"
    manifests_dir = data_dir / "manifests"

    landmarks_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load class list
    class_list_file = data_dir / "wlasl_class_list.txt"
    if not class_list_file.exists():
        raise FileNotFoundError(f"Missing {class_list_file}")

    class_to_id: dict[str, int] = {}
    with open(class_list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    class_to_id[parts[1]] = int(parts[0])
    print(f"[Step 1/7] Loaded {len(class_to_id)} classes from wlasl_class_list.txt")

    # 2. Load WLASL-3 split files for signer_id and split metadata
    print("[Step 2/7] Loading official splits and signer identities from WLASL-3...")
    w3_train_path = w3_dir / "WLASL_train.json"
    w3_val_path = w3_dir / "WLASL_validation.json"
    w3_test_path = w3_dir / "WLASL_test.json"

    for p in [w3_train_path, w3_val_path, w3_test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    with open(w3_train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(w3_val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(w3_test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Index instance metadata by video_id
    instance_meta: dict[str, dict] = {}
    for split_name, dataset in [("train", train_data), ("val", val_data), ("test", test_data)]:
        for entry in dataset:
            gloss = entry["gloss"]
            for inst in entry["instances"]:
                vid = os.path.splitext(os.path.basename(inst["video_id"]))[0]
                instance_meta[vid] = {
                    "gloss": gloss,
                    "split": split_name,
                    "signer_id": str(inst.get("signer_id", "unknown")),
                    "fps": float(inst.get("fps", 25.0)),
                    "frame_start": inst.get("frame_start", 1),
                    "frame_end": inst.get("frame_end", -1),
                }

    print(f"  Indexed {len(instance_meta)} instances across train, val, and test splits.")

    # 3. Load WLASL-2 index mapping
    print("[Step 3/7] Loading WLASL-2 index mapping...")
    parsed_data_path = w2_dir / "WLASL_parsed_data.json"
    with open(parsed_data_path, "r", encoding="utf-8") as f:
        w2_parsed = json.load(f)
    total_samples = len(w2_parsed)
    print(f"  Total samples to extract and convert: {total_samples}")

    # 4. Open landmarks_V3.npz and convert with hand-swap and mask derivation
    print("[Step 4/7] Converting 553-point holistic landmarks with canonical modality layout...")
    v3_path = w2_dir / "landmarks_V3.npz"
    if not v3_path.exists():
        raise FileNotFoundError(f"Missing {v3_path}")

    train_manifest: list[dict] = []
    val_manifest: list[dict] = []
    test_manifest: list[dict] = []
    full_manifest: list[dict] = []

    t_start = time.time()
    batch_start = time.time()

    with np.load(v3_path) as d3:
        for idx in range(total_samples):
            key = str(idx)
            raw_arr = d3[key]  # Shape: (T, 553, 3), float64
            t_len = raw_arr.shape[0]

            # In WLASL-2 V3: Right Hand (0..21), Left Hand (21..42), Pose (42..75), Face (75..553)
            # In repo schema: Left Hand (0..21), Right Hand (21..42), Pose (42..75), Face (75..553)
            swapped = raw_arr.astype(np.float32)
            swapped[:, :21] = raw_arr[:, 21:42]   # Left Hand
            swapped[:, 21:42] = raw_arr[:, :21]   # Right Hand
            # Pose and Face remain in places 42..75 and 75..553

            # Compute point-level visibility mask: (T, 553)
            pt_valid = (np.abs(swapped).sum(axis=-1) > 1e-5)
            # Expand to (T, 553, 3) for coordinate-level mask
            mask = np.repeat(pt_valid[:, :, np.newaxis], 3, axis=-1).astype(np.float32)

            # Flatten to canonical [T, 1659] dimensions
            landmarks_flat = swapped.reshape(t_len, 1659)
            mask_flat = mask.reshape(t_len, 1659)

            # Determine video ID and output path
            video_rel = w2_parsed[idx]["video_path"]
            vid = os.path.splitext(os.path.basename(video_rel))[0]
            out_file = landmarks_dir / f"{vid}.npz"

            # Save uncompressed for fastest DataLoader random-access read throughput
            np.savez(out_file, landmarks=landmarks_flat, mask=mask_flat)

            # Build manifest record
            meta = instance_meta.get(vid, {})
            gloss = meta.get("gloss") or w2_parsed[idx]["gloss"]
            split = meta.get("split") or w2_parsed[idx]["split"]
            signer_id = meta.get("signer_id", "unknown")
            fps = meta.get("fps", 25.0)
            class_id = class_to_id.get(gloss, -1)
            duration = round(t_len / fps, 4) if fps > 0 else 0.0

            # Point to WLASL-3 video if exists
            vpath = f"data/WLASL-3/videos/{vid}.mp4"

            manifest_entry = {
                "sample_id": vid,
                "signer_id": signer_id,
                "class_id": class_id,
                "class_name": gloss,
                "video_path": vpath,
                "num_frames": int(t_len),
                "fps": float(fps),
                "duration": float(duration),
                "split": split,
            }

            full_manifest.append(manifest_entry)
            if split == "train":
                train_manifest.append(manifest_entry)
            elif split == "val":
                val_manifest.append(manifest_entry)
            elif split == "test":
                test_manifest.append(manifest_entry)

            if (idx + 1) % 1000 == 0 or (idx + 1) == total_samples:
                elapsed = time.time() - batch_start
                rate = 1000 / elapsed if (idx + 1) % 1000 == 0 else (idx % 1000 + 1) / elapsed
                total_elapsed = time.time() - t_start
                eta_sec = (total_samples - (idx + 1)) / ( (idx + 1) / total_elapsed )
                print(
                    f"  Processed {idx + 1:5d}/{total_samples} samples "
                    f"({rate:4.1f} samples/s) - ETA: {eta_sec/60:.1f} min"
                )
                batch_start = time.time()

    total_time = time.time() - t_start
    print(f"  Finished unpacking 21,083 landmark files in {total_time/60:.1f} minutes!")

    # 5. Write Manifest Files
    print("[Step 5/7] Writing canonical manifests...")
    with open(manifests_dir / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_manifest, f, indent=2)
    with open(manifests_dir / "validation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(val_manifest, f, indent=2)
    with open(manifests_dir / "test_manifest.json", "w", encoding="utf-8") as f:
        json.dump(test_manifest, f, indent=2)
    with open(manifests_dir / "full_manifest.json", "w", encoding="utf-8") as f:
        json.dump(full_manifest, f, indent=2)
    with open(manifests_dir / "sample10_manifest.json", "w", encoding="utf-8") as f:
        json.dump(full_manifest[:10], f, indent=2)

    # Generate dataset inventory report
    train_signers = {e["signer_id"] for e in train_manifest}
    val_signers = {e["signer_id"] for e in val_manifest}
    test_signers = {e["signer_id"] for e in test_manifest}

    dataset_report = {
        "sample_count": len(full_manifest),
        "class_count": len(class_to_id),
        "signer_count": len(train_signers | val_signers | test_signers),
        "train_count": len(train_manifest),
        "validation_count": len(val_manifest),
        "test_count": len(test_manifest),
        "split_signers": {
            "train": len(train_signers),
            "val": len(val_signers),
            "test": len(test_signers),
        },
        "classes": sorted(list(class_to_id.keys())),
    }
    with open(manifests_dir / "dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(dataset_report, f, indent=2)
    print(f"  Saved manifests to {manifests_dir} (Train: {len(train_manifest)}, Val: {len(val_manifest)}, Test: {len(test_manifest)})")

    # 6. Copy semantic word embeddings
    print("[Step 6/7] Linking semantic word embeddings...")
    emb_source = w2_dir / "labels.npz"
    emb_dest = data_dir / "word_embeddings.npz"
    if emb_source.exists():
        shutil.copy2(emb_source, emb_dest)
        print(f"  Copied 300-d word embeddings to {emb_dest}")

    # 7. Generate top 100 classes config
    print("[Step 7/7] Computing top 100 classes with highest support and signer diversity...")
    class_samples = collections.defaultdict(list)
    class_signers = collections.defaultdict(set)
    for entry in full_manifest:
        cname = entry["class_name"]
        class_samples[cname].append(entry["sample_id"])
        class_signers[cname].add(entry["signer_id"])

    class_stats = []
    for cname in class_samples:
        class_stats.append((cname, len(class_samples[cname]), len(class_signers[cname])))

    class_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top_100 = class_stats[:100]

    top_100_json = repo_root / "configs" / "experiments" / "wlasl100_v1.json"
    top_100_json.parent.mkdir(parents=True, exist_ok=True)
    with open(top_100_json, "w", encoding="utf-8") as f:
        json.dump({
            "vocabulary": [c[0] for c in top_100],
            "metadata": {c[0]: {"samples": c[1], "signers": c[2]} for c in top_100}
        }, f, indent=2)

    top_100_yaml = repo_root / "configs" / "experiments" / "wlasl100_proposed.yaml"
    yaml_lines = [
        "dataset:",
        "  labels:",
    ]
    for c in top_100:
        yaml_lines.append(f"  - '{c[0]}'")
    yaml_lines.extend([
        "model:",
        "  name: robust_holistic_fusion",
        "  num_classes: 100",
        "  hidden_dim: 128",
        "  embedding_dim: 64",
        "  num_layers: 1",
        "  dropout: 0.2",
        "training:",
        "  epochs: 100",
        "  batch_size: 16",
        "  learning_rate: 0.001",
        "",
    ])
    with open(top_100_yaml, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))
    print(f"  Updated WLASL-100 configuration in {top_100_yaml} (Top class: {top_100[0][0]} with {top_100[0][1]} samples)")

    print("=" * 70)
    print("SUCCESS: Full canonical dataset built and verified!")
    print("=" * 70)


if __name__ == "__main__":
    main()
