from __future__ import annotations

import numpy as np

from sign_language.api.inference import RealTimePredictor, create_prediction_payload


def test_prediction_payload_and_predictor_shape() -> None:
    predictor = RealTimePredictor()
    sample = np.ones((predictor.sequence_length, predictor.input_dim), dtype=np.float32)

    payload = create_prediction_payload(sample)
    assert payload["sequence_length"] == predictor.sequence_length
    assert payload["feature_dim"] == predictor.input_dim

    logits = predictor.predict(sample)
    assert logits.shape == (predictor.num_classes,)
    assert np.isfinite(logits).all()
