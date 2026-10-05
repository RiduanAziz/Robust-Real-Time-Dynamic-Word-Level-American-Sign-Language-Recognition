from __future__ import annotations

"""Compatibility imports for the pre-canonical ``src.data`` package."""

from sign_language.data.manifest import (
    DatasetManifestEntry,
    build_manifest,
    generate_dataset_report,
    validate_signer_independent_split,
    write_manifest_json,
)

__all__ = [
    "DatasetManifestEntry",
    "build_manifest",
    "generate_dataset_report",
    "validate_signer_independent_split",
    "write_manifest_json",
]
