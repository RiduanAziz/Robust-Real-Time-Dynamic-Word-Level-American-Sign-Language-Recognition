"""Master Data Audit and Exploratory Data Analysis Script.

Executes real, empirical filesystem audits and dataset analysis on local WLASL-100 assets.
Generates:
  - results/data_audit/current_file_inventory.json
  - results/data_audit/missing_videos.csv
  - results/data_audit/split_integrity_report.json
  - results/eda/class_distribution.csv
  - results/eda/sequence_statistics.csv
  - results/eda/landmark_quality.csv
  - results/eda/feature_statistics.csv
  - results/eda/plots/ (class distribution, sequence lengths, landmark detection rates)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Ensure local src is in sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sign_language.data import CanonicalVocabulary
from sign_language.landmarks.schema import MODALITY_COORD_OFFSETS


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    manifests_dir = data_dir / "manifests"
    landmarks_dir = data_dir / "landmarks"
    audit_out_dir = root / "results" / "data_audit"
    eda_out_dir = root / "results" / "eda"
    plots_out_dir = eda_out_dir / "plots"

    audit_out_dir.mkdir(parents=True, exist_ok=True)
    plots_out_dir.mkdir(parents=True, exist_ok=True)

    print("=== SignFlow Master Data Audit and Real EDA ===")

    # 1. Load Canonical Vocabulary
    baseline_cfg_path = root / "configs" / "colab" / "wlasl100_baseline_colab.yaml"
    with open(baseline_cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    wlasl100_classes = cfg["dataset"]["labels"]
    vocab = CanonicalVocabulary(classes=wlasl100_classes)
    class_set = set(wlasl100_classes)
    print(f"Loaded {len(wlasl100_classes)} classes with vocabulary hash: {vocab.vocabulary_hash[:12]}...")

    # 2. Audit Manifests
    split_manifests = {}
    for split in ["train", "validation", "test"]:
        path = manifests_dir / f"{split}_manifest.json"
        with open(path, encoding="utf-8") as f:
            split_manifests[split] = json.load(f)
        print(f"Loaded {split} manifest: {len(split_manifests[split])} total samples")

    # Filter to WLASL-100
    wlasl100_samples = {}
    for split, samples in split_manifests.items():
        wlasl100_samples[split] = [s for s in samples if s.get("class_name") in class_set]
        print(f"  -> WLASL-100 {split}: {len(wlasl100_samples[split])} samples")

    total_wlasl100_samples = sum(len(s) for s in wlasl100_samples.values())
    print(f"Total WLASL-100 samples across splits: {total_wlasl100_samples}")

    # 3. Check Local Landmark Files
    local_npz_files = set(f.name for f in landmarks_dir.glob("*.npz"))
    print(f"Total .npz files in {landmarks_dir}: {len(local_npz_files)}")

    missing_samples = []
    available_wlasl100 = {}
    for split, samples in wlasl100_samples.items():
        avail = []
        for s in samples:
            sid = s["sample_id"]
            npz_name = f"{sid}.npz"
            if npz_name in local_npz_files:
                avail.append(s)
            else:
                missing_samples.append({
                    "sample_id": sid,
                    "split": split,
                    "class_name": s.get("class_name"),
                    "signer_id": s.get("signer_id"),
                    "video_path": s.get("video_path"),
                })
        available_wlasl100[split] = avail
        print(f"Available {split} samples with landmarks: {len(avail)} / {len(samples)}")

    # 4. Write results/data_audit/current_file_inventory.json
    inventory = {
        "vocabulary_size": len(wlasl100_classes),
        "vocabulary_hash": vocab.vocabulary_hash,
        "total_manifest_samples": sum(len(s) for s in split_manifests.values()),
        "total_wlasl100_manifest_samples": total_wlasl100_samples,
        "total_local_npz_landmarks": len(local_npz_files),
        "wlasl100_splits": {
            split: {
                "manifest_count": len(wlasl100_samples[split]),
                "available_landmark_count": len(available_wlasl100[split]),
                "missing_landmark_count": len(wlasl100_samples[split]) - len(available_wlasl100[split]),
            }
            for split in ["train", "validation", "test"]
        },
        "missing_samples_count": len(missing_samples),
    }
    with open(audit_out_dir / "current_file_inventory.json", "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    # 5. Write results/data_audit/missing_videos.csv
    with open(audit_out_dir / "missing_videos.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "split", "class_name", "signer_id", "video_path"])
        writer.writeheader()
        for item in missing_samples:
            writer.writerow(item)

    # 6. Split Integrity & Signer Overlap Analysis
    signers_per_split = {
        split: set(s["signer_id"] for s in wlasl100_samples[split] if s.get("signer_id"))
        for split in ["train", "validation", "test"]
    }
    split_integrity = {
        "protocol": "official_wlasl_benchmark_protocol_a",
        "signers_count": {split: len(signers) for split, signers in signers_per_split.items()},
        "signer_overlaps": {
            "train_val": len(signers_per_split["train"] & signers_per_split["validation"]),
            "train_test": len(signers_per_split["train"] & signers_per_split["test"]),
            "val_test": len(signers_per_split["validation"] & signers_per_split["test"]),
        },
        "signer_disjoint_compliant": (len(signers_per_split["train"] & signers_per_split["test"]) == 0),
        "note": "Official WLASL Protocol A has documented signer overlap. Protocol B enforces strict zero-leakage signer-disjoint partitioning.",
    }
    with open(audit_out_dir / "split_integrity_report.json", "w", encoding="utf-8") as f:
        json.dump(split_integrity, f, indent=2)

    # 7. Class Distribution Analysis
    class_counts = {c: {"train": 0, "validation": 0, "test": 0, "total": 0} for c in wlasl100_classes}
    for split, samples in available_wlasl100.items():
        for s in samples:
            c = s["class_name"]
            if c in class_counts:
                class_counts[c][split] += 1
                class_counts[c]["total"] += 1

    with open(eda_out_dir / "class_distribution.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class_name", "train_count", "val_count", "test_count", "total_count", "imbalance_ratio"])
        max_count = max(d["total"] for d in class_counts.values()) if class_counts else 1
        for idx, c in enumerate(wlasl100_classes):
            d = class_counts[c]
            imbalance = round(d["total"] / max_count, 4) if max_count > 0 else 0
            writer.writerow([idx, c, d["train"], d["validation"], d["test"], d["total"], imbalance])

    # 8. Sequence and Landmark Quality EDA
    print("Measuring sequence lengths, durations, and landmark detection rates across WLASL-100...")
    seq_records = []
    quality_records = []
    modality_coords = MODALITY_COORD_OFFSETS

    # Sample all available test and val, and up to 500 train for comprehensive fast coverage
    all_eval_samples = available_wlasl100["test"] + available_wlasl100["validation"] + available_wlasl100["train"][:500]

    all_l_rates, all_r_rates, all_p_rates, all_f_rates = [], [], [], []
    all_durations = []
    all_num_frames = []

    for s in all_eval_samples:
        sid = s["sample_id"]
        cname = s["class_name"]
        split = s["split"]
        npz_file = landmarks_dir / f"{sid}.npz"
        if not npz_file.exists():
            continue
        try:
            data = np.load(npz_file)
            landmarks = data["landmarks"]
            mask = data["mask"] if "mask" in data else np.ones_like(landmarks)
        except Exception:
            continue

        num_frames = landmarks.shape[0]
        fps = float(s.get("fps", 25.0) or 25.0)
        duration = num_frames / max(1.0, fps)
        all_num_frames.append(num_frames)
        all_durations.append(duration)

        # Modality detection rates
        lh_m = mask[:, modality_coords["left_hand"][0]:modality_coords["left_hand"][1]]
        rh_m = mask[:, modality_coords["right_hand"][0]:modality_coords["right_hand"][1]]
        pose_m = mask[:, modality_coords["pose"][0]:modality_coords["pose"][1]]
        face_m = mask[:, modality_coords["face"][0]:modality_coords["face"][1]]

        lh_rate = float((lh_m.sum(axis=-1) > 0).mean())
        rh_rate = float((rh_m.sum(axis=-1) > 0).mean())
        p_rate = float((pose_m.sum(axis=-1) > 0).mean())
        f_rate = float((face_m.sum(axis=-1) > 0).mean())

        all_l_rates.append(lh_rate)
        all_r_rates.append(rh_rate)
        all_p_rates.append(p_rate)
        all_f_rates.append(f_rate)

        # Consecutive missing hands
        hands_present = ((lh_m.sum(axis=-1) > 0) | (rh_m.sum(axis=-1) > 0)).astype(int)
        max_missing = 0
        cur_missing = 0
        for present in hands_present:
            if present == 0:
                cur_missing += 1
                if cur_missing > max_missing:
                    max_missing = cur_missing
            else:
                cur_missing = 0

        seq_records.append([sid, cname, split, num_frames, round(duration, 3), fps, round(float((mask.sum(axis=-1) > 0).mean()), 3)])
        quality_records.append([sid, cname, round(lh_rate, 3), round(rh_rate, 3), round(p_rate, 3), round(f_rate, 3), max_missing])

    # Write results/eda/sequence_statistics.csv
    with open(eda_out_dir / "sequence_statistics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "class_name", "split", "num_frames", "duration_sec", "fps", "valid_frame_ratio"])
        writer.writerows(seq_records)

    # Write results/eda/landmark_quality.csv
    with open(eda_out_dir / "landmark_quality.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "class_name", "lh_detection_rate", "rh_detection_rate", "pose_detection_rate", "face_detection_rate", "max_consecutive_missing_frames"])
        writer.writerows(quality_records)

    # 9. Feature Statistics EDA
    with open(eda_out_dir / "feature_statistics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["modality", "num_points", "num_coords", "mean_detection_rate", "min_detection_rate", "max_detection_rate"])
        writer.writerow(["left_hand", 21, 63, round(float(np.mean(all_l_rates)), 3), round(float(np.min(all_l_rates)), 3), round(float(np.max(all_l_rates)), 3)])
        writer.writerow(["right_hand", 21, 63, round(float(np.mean(all_r_rates)), 3), round(float(np.min(all_r_rates)), 3), round(float(np.max(all_r_rates)), 3)])
        writer.writerow(["pose", 33, 99, round(float(np.mean(all_p_rates)), 3), round(float(np.min(all_p_rates)), 3), round(float(np.max(all_p_rates)), 3)])
        writer.writerow(["face", 478, 1434, round(float(np.mean(all_f_rates)), 3), round(float(np.min(all_f_rates)), 3), round(float(np.max(all_f_rates)), 3)])

    # 10. Generate High-Quality EDA Plots
    print("Generating EDA visualizations...")

    # Plot 1: Class frequency
    plt.figure(figsize=(14, 5))
    totals = [class_counts[c]["total"] for c in wlasl100_classes]
    plt.bar(range(len(totals)), totals, color="#2563EB", alpha=0.85, width=0.8)
    plt.title("WLASL-100 Sample Frequency Across 100 Classes", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Class Index (0 - 99)", fontsize=11)
    plt.ylabel("Sample Count", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_out_dir / "class_frequency_distribution.png", dpi=150)
    plt.close()

    # Plot 2: Sequence Length Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(all_num_frames, bins=35, color="#10B981", edgecolor="black", alpha=0.85)
    plt.axvline(float(np.median(all_num_frames)), color="#DC2626", linestyle="dashed", linewidth=2, label=f"Median ({int(np.median(all_num_frames))} frames)")
    plt.axvline(float(np.mean(all_num_frames)), color="#F59E0B", linestyle="solid", linewidth=2, label=f"Mean ({np.mean(all_num_frames):.1f} frames)")
    plt.title("Sequence Frame Count Distribution (WLASL-100)", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Number of Video Frames", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_out_dir / "sequence_length_histogram.png", dpi=150)
    plt.close()

    # Plot 3: Landmark Detection Rate by Modality
    plt.figure(figsize=(8, 5))
    modalities = ["Left Hand", "Right Hand", "Pose (Body)", "Face Mesh"]
    means = [np.mean(all_l_rates) * 100, np.mean(all_r_rates) * 100, np.mean(all_p_rates) * 100, np.mean(all_f_rates) * 100]
    colors = ["#EF4444", "#3B82F6", "#8B5CF6", "#EC4899"]
    bars = plt.bar(modalities, means, color=colors, alpha=0.85, width=0.55)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.ylim(0, 110)
    plt.title("Empirical Landmark Detection Rate by Modality (MediaPipe)", fontsize=13, fontweight="bold", pad=10)
    plt.ylabel("Average Observed Frame Visibility (%)", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_out_dir / "landmark_validity_by_modality.png", dpi=150)
    plt.close()

    # Plot 4: Split Proportions
    plt.figure(figsize=(6, 6))
    split_sizes = [len(available_wlasl100["train"]), len(available_wlasl100["validation"]), len(available_wlasl100["test"])]
    split_labels = [f"Train ({split_sizes[0]})", f"Val ({split_sizes[1]})", f"Test ({split_sizes[2]})"]
    plt.pie(split_sizes, labels=split_labels, autopct="%1.1f%%", startangle=140, colors=["#3B82F6", "#F59E0B", "#10B981"], explode=(0.02, 0.02, 0.05))
    plt.title("WLASL-100 Dataset Partitioning (Protocol A)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_out_dir / "split_proportions.png", dpi=150)
    plt.close()

    print("=== Master Data Audit and Real EDA Completed Successfully! ===")


if __name__ == "__main__":
    main()
