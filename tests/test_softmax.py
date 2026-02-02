import numpy as np
from core.network import SimpleNeuralNetwork


def test_softmax_rows_sum_to_one():
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=5, output_size=3, seed=42)
    X = np.random.randn(10, 2)
    probs = nn.predict_proba(X, task="multiclass")
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-7)
