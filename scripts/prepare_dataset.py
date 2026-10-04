from __future__ import annotations

import argparse
import json

from sign_language.data import (
    DatasetValidationError,
    build_video_manifest,
    discover_video_paths,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a validated manifest from real ASL videos.")
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw video files.")
    parser.add_argument("--metadata", required=True, help="CSV with sample_id, signer_id, class_name, video_path.")
    parser.add_argument("--output", default="data/manifests/full_manifest.json")
    args = parser.parse_args()

    try:
        discovered_videos = len(discover_video_paths(args.raw_dir))
        manifest, report = build_video_manifest(args.raw_dir, args.metadata, args.output)
    except (DatasetValidationError, FileNotFoundError) as error:
        parser.error(str(error))

    print(json.dumps({"discovered_videos": discovered_videos}, indent=2))
    print(json.dumps({"manifest_path": args.output, "samples": len(manifest), "report": report.__dict__}, indent=2, default=list))


if __name__ == "__main__":
    main()
