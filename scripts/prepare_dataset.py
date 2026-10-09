from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sign_language.data import (
    DatasetValidationError,
    build_video_manifest,
    discover_video_paths,
)


def analyze_vocabulary_feasibility(manifest: list[dict], min_clips: int = 3, min_signers: int = 2) -> dict:
    """Analyze class coverage, clips per class, and independent signers."""
    class_clips: dict[str, int] = Counter()
    class_signers: dict[str, set[str]] = {}

    for entry in manifest:
        cname = entry.get("class_name", "")
        sid = entry.get("signer_id", "")
        class_clips[cname] += 1
        if cname not in class_signers:
            class_signers[cname] = set()
        if sid:
            class_signers[cname].add(sid)

    feasible_classes = []
    underrepresented_classes = []

    for cname, count in sorted(class_clips.items()):
        signers = class_signers.get(cname, set())
        info = {
            "class_name": cname,
            "available_clips": count,
            "unique_signers": len(signers),
            "is_feasible": count >= min_clips and len(signers) >= min_signers,
        }
        if info["is_feasible"]:
            feasible_classes.append(info)
        else:
            underrepresented_classes.append(info)

    return {
        "total_classes": len(class_clips),
        "feasible_class_count": len(feasible_classes),
        "underrepresented_class_count": len(underrepresented_classes),
        "feasible_classes": feasible_classes,
        "underrepresented_classes": underrepresented_classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a validated manifest from real ASL videos.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing raw video files.")
    parser.add_argument("--metadata", default="data/WLASL_v0.3.json", help="CSV or WLASL JSON metadata file.")
    parser.add_argument("--output", default="data/manifests/full_manifest.json")
    parser.add_argument("--max-samples", type=int, help="Validate only the first metadata records.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write valid records even when metadata references absent/corrupt videos.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse existing manifest if available.")
    args = parser.parse_args()

    output_path = Path(args.output)
    if args.resume and output_path.is_file():
        print(f"Resuming: Loading existing manifest from {output_path}")
        with output_path.open("r", encoding="utf-8") as handle:
            manifest_data = json.load(handle)
        feasibility = analyze_vocabulary_feasibility(manifest_data)
        print(f"Loaded {len(manifest_data)} records. Feasible classes: {feasibility['feasible_class_count']}/{feasibility['total_classes']}")
        return

    raw_path = Path(args.raw_dir)
    discovered_videos = 0
    if raw_path.exists():
        discovered_videos = len(discover_video_paths(args.raw_dir))

    try:
        manifest, report = build_video_manifest(
            args.raw_dir,
            args.metadata,
            args.output,
            max_samples=args.max_samples,
            allow_missing=args.allow_missing,
        )
    except (DatasetValidationError, FileNotFoundError) as error:
        parser.error(str(error))

    manifest_dicts = [m.__dict__ if hasattr(m, "__dict__") else m for m in manifest]
    feasibility = analyze_vocabulary_feasibility(manifest_dicts)

    print(json.dumps({"discovered_videos": discovered_videos}, indent=2))
    print(
        json.dumps(
            {
                "manifest_path": args.output,
                "valid_samples": len(manifest),
                "feasible_classes": feasibility["feasible_class_count"],
                "report_summary": {
                    "video_count": report.video_count,
                    "valid_count": report.valid_count,
                    "missing_files": len(report.missing_files),
                    "corrupt_files": len(report.corrupt_files),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
