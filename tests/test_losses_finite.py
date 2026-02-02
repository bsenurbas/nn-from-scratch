import numpy as np
from core.losses import bce, cross_entropy


def test_bce_is_finite():
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    p = np.array([[0.1], [0.9], [0.8], [0.2]])
    loss = bce(y, p)
    assert np.isfinite(loss)


def test_cross_entropy_is_finite():
    y_onehot = np.eye(3)[np.array([0, 1, 2, 1])]
    p = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
        [0.2, 0.6, 0.2],
    ])
    loss = cross_entropy(y_onehot, p)
    assert np.isfinite(loss)
