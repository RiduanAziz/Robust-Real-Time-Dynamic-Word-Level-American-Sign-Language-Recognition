"""Baseline and environment audit script."""
import os
import sys
import json
import platform
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

def run_audit():
    repo_root = Path(__file__).resolve().parent.parent
    report = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "repo_root": str(repo_root),
    }

    # PyTorch and CUDA
    try:
        import torch
        report["torch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            report["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        report["torch_error"] = str(e)

    # MediaPipe
    try:
        import mediapipe as mp
        report["mediapipe_version"] = mp.__version__
    except Exception as e:
        report["mediapipe_error"] = str(e)

    # Checkpoints in models/
    models_dir = repo_root / "models"
    checkpoints = []
    if models_dir.exists():
        for f in models_dir.rglob("*"):
            if f.is_file():
                checkpoints.append({
                    "rel_path": str(f.relative_to(repo_root)),
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2)
                })
    report["checkpoints"] = checkpoints

    # Manifests in data/manifests/
    manifests_dir = repo_root / "data" / "manifests"
    manifest_info = {}
    if manifests_dir.exists():
        for mf in manifests_dir.glob("*.json"):
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        manifest_info[mf.name] = {
                            "type": "list",
                            "count": len(data),
                            "sample_keys": list(data[0].keys()) if data else []
                        }
                    elif isinstance(data, dict):
                        manifest_info[mf.name] = {
                            "type": "dict",
                            "keys": list(data.keys())[:10],
                            "sample_count": len(data.get("samples", [])) if "samples" in data else len(data)
                        }
            except Exception as e:
                manifest_info[mf.name] = {"error": str(e)}
    report["manifests"] = manifest_info

    # Data folders (shallow count)
    data_dir = repo_root / "data"
    data_structure = {}
    if data_dir.exists():
        for child in data_dir.iterdir():
            if child.is_dir():
                try:
                    entries = list(child.iterdir())
                    data_structure[child.name] = {
                        "is_dir": True,
                        "direct_children": len(entries)
                    }
                except Exception as e:
                    data_structure[child.name] = {"error": str(e)}
            else:
                data_structure[child.name] = {
                    "is_dir": False,
                    "size_kb": round(child.stat().st_size / 1024, 2)
                }
    report["data_structure"] = data_structure

    # Results directory
    results_dir = repo_root / "results"
    results_files = []
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            results_files.append({
                "name": f.name,
                "size_kb": round(f.stat().st_size / 1024, 2)
            })
    report["results"] = results_files

    output_path = repo_root / "results" / "baseline_environment_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("AUDIT_COMPLETE")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    run_audit()
