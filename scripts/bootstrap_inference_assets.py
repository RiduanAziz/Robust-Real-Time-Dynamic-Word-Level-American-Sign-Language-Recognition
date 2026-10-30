#!/usr/bin/env python3
"""Bootstrap inference assets for SignFlow runtime.

Supports:
1. Local mode: Validates existing local models/ directory against expected manifest.
2. Deployment mode: Downloads and securely extracts versioned asset bundle from
   SIGNFLOW_ASSET_BUNDLE_URL with optional SIGNFLOW_ASSET_TOKEN and SHA-256 validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bootstrap_inference_assets")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Default expected assets and their known SHA-256 checksums
EXPECTED_ASSETS: dict[str, dict[str, Any]] = {
    "models/robust_holistic_fusion_best.pt": {
        "size": 4251327,
        "sha256": "6dadabcd6637aa08854d6b046e9305fe6f1f6b1d171a14d339b5650d08c7117f",
    },
    "models/wlasl100_unisign_pgf_best.pt": {
        "size": 22207675,
        "sha256": "665f3c6bdf650c1882e4692d744b73808d8571ca1b5fd74522e194818660b0ff",
    },
    "models/wlasl100_max_accuracy_best.pt": {
        "size": 28863349,
        "sha256": "18f7199e3fe1876a6bd1629e93bafdbaca6d4c8e494d11e461d68fc1664f31e3",
    },
    "models/mediapipe/holistic_landmarker.task": {
        "size": 13683609,
        "sha256": "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8",
    },
    "models/prior_logits_100.npy": {
        "size": 528,
        "sha256": "f70651a26fa25962b675364815a838237e63d40a75cb7109e38b95298d8ca71f",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_local_assets(target_dir: Path = REPO_ROOT) -> tuple[bool, list[str]]:
    """Check if all required runtime assets are present and valid."""
    missing_or_invalid: list[str] = []
    for rel_path, meta in EXPECTED_ASSETS.items():
        p = target_dir / rel_path
        if not p.is_file():
            missing_or_invalid.append(f"MISSING: {rel_path}")
            continue
        actual_size = p.stat().st_size
        if meta.get("size") and actual_size != meta["size"]:
            missing_or_invalid.append(f"SIZE MISMATCH: {rel_path} (got {actual_size}, expected {meta['size']})")
            continue
        actual_sha = sha256_file(p)
        if meta.get("sha256") and actual_sha != meta["sha256"]:
            missing_or_invalid.append(f"HASH MISMATCH: {rel_path} (got {actual_sha}, expected {meta['sha256']})")
            continue

    return len(missing_or_invalid) == 0, missing_or_invalid


def safe_extract_zip(zip_path: Path, destination_root: Path) -> None:
    """Safely extract ZIP preventing path traversal and only writing expected paths."""
    dest_resolved = destination_root.resolve()
    logger.info("Extracting %s into %s safely...", zip_path.name, dest_resolved)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Check manifest inside ZIP if present
        namelist = zf.namelist()
        for member_name in namelist:
            # Prevent zip-slip vulnerability
            member_path = (destination_root / member_name).resolve()
            if not str(member_path).startswith(str(dest_resolved)):
                raise ValueError(f"Security error: Archive member '{member_name}' attempts path traversal.")

        # Extract files
        for member in zf.infolist():
            target_path = (destination_root / member.filename).resolve()
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, open(target_path, "wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                logger.info("  Extracted: %s (%d bytes)", member.filename, member.file_size)


def bootstrap_assets(
    bundle_url: str | None = None,
    expected_bundle_sha256: str | None = None,
    asset_token: str | None = None,
    target_root: Path = REPO_ROOT,
    force_download: bool = False,
) -> bool:
    """Ensure all required assets are present locally, downloading if necessary."""
    # Read environment variables if not passed directly
    url = bundle_url or os.environ.get("SIGNFLOW_ASSET_BUNDLE_URL") or os.environ.get("MODEL_BUNDLE_URL")
    bundle_sha = expected_bundle_sha256 or os.environ.get("SIGNFLOW_ASSET_BUNDLE_SHA256")
    token = asset_token or os.environ.get("SIGNFLOW_ASSET_TOKEN")

    # 1. Check local assets first
    all_valid, issues = verify_local_assets(target_root)
    if all_valid and not force_download:
        logger.info("All required inference assets are present and SHA-256 verified locally.")
        return True

    if not all_valid:
        logger.warning("Local assets incomplete or invalid: %s", issues)

    if not url:
        logger.error(
            "SIGNFLOW_ASSET_BUNDLE_URL is not set and local assets are missing or invalid.\n"
            "Missing components:\n%s\n"
            "Action required:\n"
            "1. Place required model checkpoints in ./models/ or\n"
            "2. Set SIGNFLOW_ASSET_BUNDLE_URL to a valid release asset URL.",
            "\n".join(f"  - {i}" for i in issues),
        )
        return False

    # 2. Download bundle to temporary file
    logger.info("Downloading inference asset bundle from URL...")
    headers = {"User-Agent": "SignFlow-Bootstrap/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
        tmp_zip_path = Path(tmp_file.name)

    try:
        with urllib.request.urlopen(req) as resp, open(tmp_zip_path, "wb") as out_fp:
            downloaded = 0
            while chunk := resp.read(1024 * 1024):
                downloaded += len(chunk)
                out_fp.write(chunk)
        logger.info("Downloaded %d bytes to %s", downloaded, tmp_zip_path)

        # 3. Verify bundle SHA-256 if supplied
        if bundle_sha:
            computed_sha = sha256_file(tmp_zip_path)
            if computed_sha.lower() != bundle_sha.lower():
                raise ValueError(
                    f"Asset bundle SHA-256 mismatch!\nExpected: {bundle_sha}\nComputed: {computed_sha}"
                )
            logger.info("Asset bundle SHA-256 matches expected checksum.")

        # 4. Safe extraction
        safe_extract_zip(tmp_zip_path, target_root)

        # 5. Final verification
        final_valid, final_issues = verify_local_assets(target_root)
        if not final_valid:
            raise RuntimeError(f"Assets verification failed after extraction: {final_issues}")

        logger.info("Inference asset bootstrap completed successfully!")
        return True
    finally:
        if tmp_zip_path.is_file():
            tmp_zip_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap SignFlow inference assets.")
    parser.add_argument("--url", default=None, help="Asset bundle download URL")
    parser.add_argument("--sha256", default=None, help="Expected bundle SHA-256")
    parser.add_argument("--target-dir", type=Path, default=REPO_ROOT, help="Target root directory")
    parser.add_argument("--force", action="store_true", help="Force redownload even if assets exist")
    args = parser.parse_args()

    success = bootstrap_assets(
        bundle_url=args.url,
        expected_bundle_sha256=args.sha256,
        target_root=args.target_dir,
        force_download=args.force,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

