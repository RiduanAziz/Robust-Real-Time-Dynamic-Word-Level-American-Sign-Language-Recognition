import torch
from sign_language.config.loader import load_experiment_config
from sign_language.models.factory import build_model

config = load_experiment_config("configs/experiments/wlasl100_proposed.yaml")
model = build_model(config)

dummy_input = torch.randn(2, config.model.sequence_length, config.model.input_dim)
logits = model(dummy_input)

print(f"Model initialized successfully. Output shape: {logits.shape}")
assert logits.shape == (2, 100), f"Expected (2, 100), got {logits.shape}"
print("Pipeline components successfully support 100 classes.")
