from __future__ import annotations

import argparse

import numpy as np

from sign_language.landmarks import LandmarkExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract synthetic landmark features from a frame.")
    parser.add_argument("--feature-dim", type=int, default=42)
    args = parser.parse_args()

    extractor = LandmarkExtractor(feature_dim=args.feature_dim)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    features = extractor.extract(frame)
    print(f"Extracted landmark shape: {features.shape}")


if __name__ == "__main__":
    main()
