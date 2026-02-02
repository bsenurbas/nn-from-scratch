import numpy as np
from core.network import SimpleNeuralNetwork
from core.utils import one_hot


def test_score_binary_accepts_1d_and_2d_y():
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=3, output_size=1, seed=42)
    X = np.random.randn(6, 2)

    # Fake labels
    y1 = np.array([0, 1, 0, 1, 0, 1])       # (n,)
    y2 = y1.reshape(-1, 1).astype(float)    # (n,1)

    # Just check it runs and returns float
    s1 = nn.score(X, y1, task="binary")
    s2 = nn.score(X, y2, task="binary")
    assert isinstance(s1, float)
    assert isinstance(s2, float)


def test_evaluate_multiclass_accepts_ids_and_onehot():
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=3, seed=42)
    X = np.random.randn(8, 2)
    y_ids = np.array([0, 1, 2, 1, 0, 2, 1, 0])
    y_oh = one_hot(y_ids, num_classes=3)

    out1 = nn.evaluate(X, y_ids, task="multiclass")
    out2 = nn.evaluate(X, y_oh, task="multiclass")

    assert "loss" in out1 and "acc" in out1
    assert "loss" in out2 and "acc" in out2
    assert np.isfinite(out1["loss"])
    assert np.isfinite(out2["loss"])
