"""Dataset audit script: verifies signer overlap, sample integrity, and vocabulary alignment."""
import json
from pathlib import Path
from collections import Counter
import yaml

def run_dataset_audit():
    repo_root = Path(__file__).resolve().parent.parent
    manifests_dir = repo_root / "data" / "manifests"

    train_path = manifests_dir / "train_manifest.json"
    val_path = manifests_dir / "validation_manifest.json"
    test_path = manifests_dir / "test_manifest.json"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }

    report = {}

    # Sample counts, signers, classes
    for name, data in splits.items():
        sample_ids = [d["sample_id"] for d in data]
        signer_ids = [d.get("signer_id", "UNKNOWN") for d in data]
        class_names = [d["class_name"] for d in data]
        class_ids = [d["class_id"] for d in data]

        unique_samples = set(sample_ids)
        unique_signers = set(signer_ids)
        unique_classes = set(class_names)

        report[name] = {
            "total_samples": len(data),
            "unique_sample_ids": len(unique_samples),
            "unique_signers": len(unique_signers),
            "unique_classes": len(unique_classes),
            "duplicate_samples": len(data) - len(unique_samples),
            "signers_list": sorted(list(unique_signers))
        }

    # Pairwise overlaps
    train_signers = set(report["train"]["signers_list"])
    val_signers = set(report["val"]["signers_list"])
    test_signers = set(report["test"]["signers_list"])

    train_val_signer_overlap = train_signers.intersection(val_signers)
    train_test_signer_overlap = train_signers.intersection(test_signers)
    val_test_signer_overlap = val_signers.intersection(test_signers)

    report["signer_overlap"] = {
        "train_val_overlap_count": len(train_val_signer_overlap),
        "train_test_overlap_count": len(train_test_signer_overlap),
        "val_test_overlap_count": len(val_test_signer_overlap),
        "train_test_overlap_signers": sorted(list(train_test_signer_overlap))[:15],
    }

    # Pairwise sample ID overlap
    train_samples = set(d["sample_id"] for d in train_data)
    val_samples = set(d["sample_id"] for d in val_data)
    test_samples = set(d["sample_id"] for d in test_data)

    report["sample_id_overlap"] = {
        "train_val_sample_overlap": len(train_samples.intersection(val_samples)),
        "train_test_sample_overlap": len(train_samples.intersection(test_samples)),
        "val_test_sample_overlap": len(val_samples.intersection(test_samples)),
    }

    # Landmark files existence check
    landmarks_dir = repo_root / "data" / "landmarks"
    missing_landmarks = {}
    for name, data in splits.items():
        missing_count = 0
        missing_samples = []
        for d in data:
            sid = d["sample_id"]
            npz_file = landmarks_dir / f"{sid}.npz"
            if not npz_file.is_file():
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(sid)
        missing_landmarks[name] = {
            "missing_count": missing_count,
            "sample_missing": missing_samples
        }
    report["missing_landmarks"] = missing_landmarks

    # Vocabulary checks across configs and reports
    vocab_info = {}
    config_yaml = repo_root / "configs" / "experiments" / "wlasl100_proposed.yaml"
    if config_yaml.is_file():
        with open(config_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            labels = cfg.get("dataset", {}).get("labels", [])
            vocab_info["wlasl100_proposed_yaml_labels_count"] = len(labels)
            vocab_info["wlasl100_proposed_yaml_sample"] = labels[:5]

    wlasl100_json = repo_root / "configs" / "experiments" / "wlasl100_v1.json"
    if wlasl100_json.is_file():
        with open(wlasl100_json, "r", encoding="utf-8") as f:
            v_data = json.load(f)
            vocab_info["wlasl100_v1_json_keys"] = list(v_data.keys())[:5]
            if isinstance(v_data, list):
                vocab_info["wlasl100_v1_json_count"] = len(v_data)

    eval_report_path = repo_root / "results" / "wlasl100_evaluation_report.json"
    if eval_report_path.is_file():
        with open(eval_report_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            vocab_info["eval_report_classes_count"] = len(eval_data.get("per_class_metrics", {}))
            vocab_info["eval_report_top1"] = eval_data.get("top_1_accuracy")
            vocab_info["eval_report_total_samples"] = eval_data.get("total_samples")

    report["vocab_and_eval"] = vocab_info

    out_file = repo_root / "results" / "dataset_leakage_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("DATASET_AUDIT_COMPLETE")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    run_dataset_audit()
