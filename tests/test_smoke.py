import numpy as np
import pandas as pd

from src.perceived_safety_model import (
    CumulativeOrderedProbit,
    make_features,
)


def test_feature_vector_uses_actual_motion_speed():
    frame = pd.DataFrame(
        {
            "speed_scale": [0.3, 0.6],
            "distance_m": [0.2, 0.8],
        }
    )
    X = make_features(frame)
    np.testing.assert_allclose(X[0], [0.45, 0.2, 0.04, 0.09])
    np.testing.assert_allclose(X[1], [0.90, 0.8, 0.64, 0.72])


def test_ordered_probit_returns_valid_probabilities():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 4))
    y = np.clip(np.rint(5 - X[:, 0] + X[:, 1] + rng.normal(size=80)), 1, 10).astype(int) - 1
    model = CumulativeOrderedProbit().fit(X, y)
    probabilities = model.predict_proba(X[:5])
    assert probabilities.shape == (5, 10)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
