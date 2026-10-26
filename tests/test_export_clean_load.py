"""Clean-load verification test for exported model bundle."""
import json
from pathlib import Path
import pytest
import torch

from sign_language.config import load_experiment_config
from sign_language.data.vocabulary import CanonicalVocabulary
from sign_language.models.factory import build_model


def test_exported_bundle_clean_load():
    export_dir = Path("outputs/final")
    if not export_dir.is_dir():
        pytest.skip("outputs/final directory not found")

    ckpt_path = export_dir / "model_best.pt"
    config_path = export_dir / "config.yaml"
    vocab_path = export_dir / "vocabulary.json"
    provenance_path = export_dir / "provenance.json"

    assert ckpt_path.is_file(), "Exported checkpoint missing"
    assert config_path.is_file(), "Exported config missing"
    assert vocab_path.is_file(), "Exported vocabulary missing"
    assert provenance_path.is_file(), "Exported provenance missing"

    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    classes = vocab_data["vocabulary"]
    vocab = CanonicalVocabulary(classes)
    assert vocab.vocabulary_hash == vocab_data["vocabulary_hash"]

    # Load config and model
    cfg = load_experiment_config(str(config_path))
    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Verify forward pass on fixed input
    torch.manual_seed(42)
    dummy_input = torch.randn(1, cfg.dataset.sequence_length, cfg.model.input_dim)
    dummy_mask = torch.ones(1, cfg.dataset.sequence_length)
    with torch.no_grad():
        out = model(dummy_input, mask=dummy_mask)

    assert out.shape == (1, len(classes)), f"Output shape mismatch: {out.shape} vs {(1, len(classes))}"
    assert torch.isfinite(out).all(), "Non-finite logits produced"
