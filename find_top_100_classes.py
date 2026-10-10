import json
import collections
from pathlib import Path

# Load manifest to get all available landmarks
manifest_path = Path("data/manifests/full_manifest.json")
with open(manifest_path, "r") as f:
    entries = json.load(f)

# Count samples and unique signers per class
class_samples = collections.defaultdict(list)
class_signers = collections.defaultdict(set)

for entry in entries:
    cname = entry["class_name"]
    class_samples[cname].append(entry["sample_id"])
    class_signers[cname].add(entry["signer_id"])

# Score classes based on (num_samples, num_signers)
class_stats = []
for cname in class_samples:
    num_samples = len(class_samples[cname])
    num_signers = len(class_signers[cname])
    class_stats.append((cname, num_samples, num_signers))

# Sort by num_samples descending, then num_signers descending
class_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)

top_100 = class_stats[:100]
print("Top 100 classes:")
for i, (cname, num_samples, num_signers) in enumerate(top_100):
    print(f"{i+1:3d}. {cname:20s}: {num_samples:3d} samples, {num_signers:2d} signers")

# Write to configs/experiments/wlasl100_v1.json
out_path = Path("configs/experiments/wlasl100_v1.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump({
        "vocabulary": [c[0] for c in top_100],
        "metadata": {
            c[0]: {"samples": c[1], "signers": c[2]} for c in top_100
        }
    }, f, indent=2)

print(f"\\nWrote top 100 classes to {out_path}")
