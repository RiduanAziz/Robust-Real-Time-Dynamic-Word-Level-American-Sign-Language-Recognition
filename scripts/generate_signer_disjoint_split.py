"""Generate a rigorous, reproducible signer-disjoint split (Protocol B)."""
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

def build_signer_disjoint_split():
    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = repo_root / "data" / "manifests" / "full_manifest.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        full_manifest = json.load(f)

    # Group samples by signer
    signer_samples = defaultdict(list)
    signer_classes = defaultdict(set)
    for sample in full_manifest:
        signer_id = str(sample.get("signer_id", "UNKNOWN"))
        signer_samples[signer_id].append(sample)
        signer_classes[signer_id].add(sample["class_name"])

    all_signers = sorted(list(signer_samples.keys()))
    print(f"Total samples: {len(full_manifest)}, Total unique signers: {len(all_signers)}")

    # Deterministic seed
    rng = random.Random(42)

    # Compute total samples per signer
    signer_counts = {s: len(signer_samples[s]) for s in all_signers}
    total_samples = len(full_manifest)
    target_train = int(total_samples * 0.70)
    target_val = int(total_samples * 0.15)
    target_test = total_samples - target_train - target_val

    # Sort signers by number of samples descending to bin-pack effectively
    sorted_signers = sorted(all_signers, key=lambda s: (len(signer_samples[s]), len(signer_classes[s])), reverse=True)

    train_signers = set()
    val_signers = set()
    test_signers = set()

    train_count = 0
    val_count = 0
    test_count = 0

    # Stratified/balanced assignment:
    # Ensure train gets ~70%, val ~15%, test ~15%
    for signer in sorted_signers:
        cnt = signer_counts[signer]
        # Decide which bucket is furthest below target ratio
        deficit_train = (target_train - train_count) / target_train
        deficit_val = (target_val - val_count) / target_val
        deficit_test = (target_test - test_count) / target_test

        if deficit_train >= deficit_val and deficit_train >= deficit_test:
            train_signers.add(signer)
            train_count += cnt
        elif deficit_val >= deficit_test:
            val_signers.add(signer)
            val_count += cnt
        else:
            test_signers.add(signer)
            test_count += cnt

    # Validate 0 overlap
    assert len(train_signers & val_signers) == 0, "Signer overlap detected between train and val!"
    assert len(train_signers & test_signers) == 0, "Signer overlap detected between train and test!"
    assert len(val_signers & test_signers) == 0, "Signer overlap detected between val and test!"
    assert len(train_signers | val_signers | test_signers) == len(all_signers), "Not all signers assigned!"

    train_manifest = []
    val_manifest = []
    test_manifest = []

    for sample in full_manifest:
        s = sample.copy()
        sid = str(s["signer_id"])
        if sid in train_signers:
            s["split"] = "train"
            train_manifest.append(s)
        elif sid in val_signers:
            s["split"] = "validation"
            val_manifest.append(s)
        elif sid in test_signers:
            s["split"] = "test"
            test_manifest.append(s)
        else:
            raise ValueError(f"Unassigned signer {sid}")

    out_dir = repo_root / "data" / "manifests" / "signer_disjoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_manifest, f, indent=2)
    with open(out_dir / "validation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(val_manifest, f, indent=2)
    with open(out_dir / "test_manifest.json", "w", encoding="utf-8") as f:
        json.dump(test_manifest, f, indent=2)

    train_classes = set(s["class_name"] for s in train_manifest)
    val_classes = set(s["class_name"] for s in val_manifest)
    test_classes = set(s["class_name"] for s in test_manifest)

    # Classes in 100-subset
    cfg_100_path = repo_root / "configs" / "experiments" / "wlasl100_proposed.yaml"
    import yaml
    with open(cfg_100_path, "r", encoding="utf-8") as f:
        cfg_100 = yaml.safe_load(f)
    wlasl100_labels = set(cfg_100["dataset"]["labels"])

    report = {
        "protocol": "Protocol B — Signer-disjoint",
        "total_signers": len(all_signers),
        "split_signers": {
            "train": len(train_signers),
            "validation": len(val_signers),
            "test": len(test_signers),
        },
        "sample_counts": {
            "train": len(train_manifest),
            "validation": len(val_manifest),
            "test": len(test_manifest),
            "total": len(train_manifest) + len(val_manifest) + len(test_manifest),
        },
        "class_coverage_2000": {
            "train": len(train_classes),
            "validation": len(val_classes),
            "test": len(test_classes),
            "all_3_splits": len(train_classes & val_classes & test_classes),
        },
        "class_coverage_wlasl100": {
            "train": len(train_classes & wlasl100_labels),
            "validation": len(val_classes & wlasl100_labels),
            "test": len(test_classes & wlasl100_labels),
            "all_3_splits": len(train_classes & val_classes & test_classes & wlasl100_labels),
        },
        "signer_overlap": {
            "train_val": len(train_signers & val_signers),
            "train_test": len(train_signers & test_signers),
            "val_test": len(val_signers & test_signers),
        }
    }

    report_path = out_dir / "signer_disjoint_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("SIGNER_DISJOINT_SPLIT_CREATED")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    build_signer_disjoint_split()
