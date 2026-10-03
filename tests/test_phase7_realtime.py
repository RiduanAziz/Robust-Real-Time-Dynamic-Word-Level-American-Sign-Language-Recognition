from __future__ import annotations

import numpy as np

from sign_language.api.inference import RealTimePredictor, create_prediction_payload


def test_prediction_payload_and_predictor_shape() -> None:
    predictor = RealTimePredictor(num_classes=5)
    sample = np.ones((16, 42), dtype=np.float32)

    payload = create_prediction_payload(sample)
    assert payload["sequence_length"] == 16
    assert payload["feature_dim"] == 42

    logits = predictor.predict(sample)
    assert logits.shape == (5,)
    assert np.isfinite(logits).all()
