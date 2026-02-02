import numpy as np
from core.network import SimpleNeuralNetwork


def test_forward_binary_shape():
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=3, output_size=1, seed=42)
    X = np.random.randn(5, 2)
    probs = nn.predict_proba(X, task="binary")
    assert probs.shape == (5, 1)


def test_forward_multiclass_shape():
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=3, output_size=3, seed=42)
    X = np.random.randn(5, 2)
    probs = nn.predict_proba(X, task="multiclass")
    assert probs.shape == (5, 3)
