import json
import yaml
from pathlib import Path

# Load wlasl100_v1.json
json_path = Path("configs/experiments/wlasl100_v1.json")
with open(json_path, "r") as f:
    data = json.load(f)

vocab = data["vocabulary"]

# Create wlasl100_proposed.yaml
config = {
    "dataset": {
        "labels": vocab
    },
    "model": {
        "name": "robust_holistic_fusion",
        "num_classes": 100,
        "hidden_dim": 128,
        "embedding_dim": 64,
        "num_layers": 1,
        "dropout": 0.2
    },
    "training": {
        "epochs": 15,
        "batch_size": 16,
        "learning_rate": 0.001
    }
}

out_yaml = Path("configs/experiments/wlasl100_proposed.yaml")
with open(out_yaml, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Generated {out_yaml}")
