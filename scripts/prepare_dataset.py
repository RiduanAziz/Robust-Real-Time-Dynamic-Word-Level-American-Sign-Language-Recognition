from __future__ import annotations

from sign_language.data import build_synthetic_dataset


def main() -> None:
    samples = build_synthetic_dataset(samples_per_class=4, sequence_length=16, feature_dim=42)
    print(f"Prepared {len(samples)} synthetic sign samples.")
    print({"labels": sorted({sample.label for sample in samples}), "signers": sorted({sample.signer_id for sample in samples})[:5]})


if __name__ == "__main__":
    main()
