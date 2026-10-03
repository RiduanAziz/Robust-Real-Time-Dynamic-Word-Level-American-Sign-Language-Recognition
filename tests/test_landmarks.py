from __future__ import annotations

import numpy as np

from sign_language.landmarks import LandmarkExtractor


def test_landmark_extractor_shapes() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    points = LandmarkExtractor(feature_dim=42).extract(frame)
    assert points.shape == (42,)
    assert np.isfinite(points).all()
