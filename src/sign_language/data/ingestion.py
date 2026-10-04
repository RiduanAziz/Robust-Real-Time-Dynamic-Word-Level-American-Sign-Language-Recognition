from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import DatasetManifestEntry, build_manifest, write_manifest_json

REQUIRED_METADATA_COLUMNS = {"sample_id", "signer_id", "class_name", "video_path"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".webm", ".mkv"}


class DatasetValidationError(ValueError):
    """Raised when source metadata cannot produce a valid dataset manifest."""


@dataclass(frozen=True)
class DatasetValidationReport:
    video_count: int
    valid_count: int
    missing_files: tuple[str, ...]
    corrupt_files: tuple[str, ...]
    duplicate_sample_ids: tuple[str, ...]
    duplicate_video_paths: tuple[str, ...]
    missing_signer_ids: tuple[str, ...]
    missing_labels: tuple[str, ...]
    invalid_fps: tuple[str, ...]
    invalid_frame_counts: tuple[str, ...]
    class_distribution: dict[str, int]
    signer_distribution: dict[str, int]

    @property
    def is_valid(self) -> bool:
        return not any(
            (
                self.missing_files,
                self.corrupt_files,
                self.duplicate_sample_ids,
                self.duplicate_video_paths,
                self.missing_signer_ids,
                self.missing_labels,
                self.invalid_fps,
                self.invalid_frame_counts,
            )
        )


def discover_video_paths(raw_dir: str | Path) -> list[Path]:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def read_metadata(path: str | Path) -> list[dict[str, str]]:
    metadata_path = Path(path)
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise DatasetValidationError(f"Metadata file contains no records: {metadata_path}")
    columns = set(rows[0].keys())
    missing_columns = sorted(REQUIRED_METADATA_COLUMNS.difference(columns))
    if missing_columns:
        raise DatasetValidationError(f"Metadata is missing required columns: {missing_columns}")
    return [{str(key): str(value or "").strip() for key, value in row.items()} for row in rows]


def _video_stats(path: Path) -> tuple[int, float, float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise DatasetValidationError(f"Unable to open video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0 or fps <= 0:
            raise DatasetValidationError(f"Video has invalid frame metadata: {path}")
        return frame_count, fps, frame_count / fps
    finally:
        capture.release()


def validate_metadata_rows(rows: Iterable[Mapping[str, Any]]) -> DatasetValidationReport:
    rows = list(rows)
    sample_ids = [str(row.get("sample_id", "")).strip() for row in rows]
    video_paths = [str(row.get("video_path", "")).strip() for row in rows]
    labels = [str(row.get("class_name", "")).strip() for row in rows]
    signers = [str(row.get("signer_id", "")).strip() for row in rows]

    def duplicates(values: list[str]) -> tuple[str, ...]:
        return tuple(sorted(value for value, count in Counter(values).items() if value and count > 1))

    missing_signers = tuple(str(index) for index, value in enumerate(signers) if not value)
    missing_labels = tuple(str(index) for index, value in enumerate(labels) if not value)
    class_distribution = Counter(label for label in labels if label)
    signer_distribution = Counter(signer for signer in signers if signer)
    return DatasetValidationReport(
        video_count=len(rows),
        valid_count=0,
        missing_files=(),
        corrupt_files=(),
        duplicate_sample_ids=duplicates(sample_ids),
        duplicate_video_paths=duplicates(video_paths),
        missing_signer_ids=missing_signers,
        missing_labels=missing_labels,
        invalid_fps=(),
        invalid_frame_counts=(),
        class_distribution=dict(sorted(class_distribution.items())),
        signer_distribution=dict(sorted(signer_distribution.items())),
    )


def build_video_manifest(
    raw_dir: str | Path,
    metadata_path: str | Path,
    output_path: str | Path | None = None,
) -> tuple[list[DatasetManifestEntry], DatasetValidationReport]:
    raw_root = Path(raw_dir).resolve()
    rows = read_metadata(metadata_path)
    preliminary = validate_metadata_rows(rows)
    missing_files: list[str] = []
    corrupt_files: list[str] = []
    invalid_fps: list[str] = []
    invalid_frame_counts: list[str] = []
    entries: list[dict[str, Any]] = []

    for row in rows:
        video_path = (raw_root / row["video_path"]).resolve()
        if not video_path.is_file():
            missing_files.append(row["video_path"])
            continue
        try:
            num_frames, fps, duration = _video_stats(video_path)
        except DatasetValidationError:
            corrupt_files.append(row["video_path"])
            continue
        if fps <= 0:
            invalid_fps.append(row["video_path"])
        if num_frames <= 0:
            invalid_frame_counts.append(row["video_path"])
        entries.append(
            {
                "sample_id": row["sample_id"],
                "signer_id": row["signer_id"],
                "class_id": int(row.get("class_id") or 0),
                "class_name": row["class_name"],
                "video_path": str(video_path.relative_to(raw_root)),
                "num_frames": num_frames,
                "fps": fps,
                "duration": duration,
            }
        )

    report = DatasetValidationReport(
        video_count=preliminary.video_count,
        valid_count=len(entries),
        missing_files=tuple(sorted(missing_files)),
        corrupt_files=tuple(sorted(corrupt_files)),
        duplicate_sample_ids=preliminary.duplicate_sample_ids,
        duplicate_video_paths=preliminary.duplicate_video_paths,
        missing_signer_ids=preliminary.missing_signer_ids,
        missing_labels=preliminary.missing_labels,
        invalid_fps=tuple(sorted(invalid_fps)),
        invalid_frame_counts=tuple(sorted(invalid_frame_counts)),
        class_distribution=preliminary.class_distribution,
        signer_distribution=preliminary.signer_distribution,
    )
    if not report.is_valid:
        raise DatasetValidationError(report)

    manifest = build_manifest(entries)
    if output_path is not None:
        write_manifest_json(manifest, output_path)
    return manifest, report