from __future__ import annotations

import argparse
import json
from pathlib import Path

from sign_language.data import (
    build_manifest,
    generate_dataset_report,
    split_manifest_by_signer,
    validate_signer_independent_split,
    write_manifest_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic signer-independent manifests.")
    parser.add_argument("--manifest", default="data/manifests/full_manifest.json")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        manifest = build_manifest(json.load(handle))
    train, validation, test = split_manifest_by_signer(manifest, seed=args.seed)
    if not validate_signer_independent_split(train, validation, test):
        raise ValueError("Generated manifests are not signer-independent")
    outputs = write_manifest_splits(train, validation, test, args.output_dir)
    report = generate_dataset_report([*train, *validation, *test])
    report["seed"] = args.seed
    report["split_signers"] = {
        "train": sorted({item.signer_id for item in train}),
        "validation": sorted({item.signer_id for item in validation}),
        "test": sorted({item.signer_id for item in test}),
    }
    report_path = Path(args.output_dir) / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": {name: str(path) for name, path in outputs.items()}, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()