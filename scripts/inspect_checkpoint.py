"""Inspect model checkpoint metadata and keys."""
import torch
from pathlib import Path

ckpt_path = Path("models/robust_holistic_fusion_best.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("Checkpoint Type:", type(ckpt))
if isinstance(ckpt, dict):
    print("Keys in checkpoint:", list(ckpt.keys()))
    for k in ckpt.keys():
        if k == "model_state_dict":
            print("  model_state_dict has", len(ckpt[k]), "tensors")
            # Print sample layer shapes
            sample_keys = list(ckpt[k].keys())[:5]
            for sk in sample_keys:
                print(f"    {sk}: {ckpt[k][sk].shape}")
            # Print classifier weight shape
            last_keys = list(ckpt[k].keys())[-3:]
            for lk in last_keys:
                print(f"    {lk}: {ckpt[k][lk].shape}")
        elif k in ("class_names", "classes", "vocabulary", "labels"):
            print(f"  {k}: length={len(ckpt[k])}, sample={ckpt[k][:5]}")
        elif k == "config":
            print(f"  config: {ckpt[k]}")
        else:
            print(f"  {k}: {ckpt[k]}")
