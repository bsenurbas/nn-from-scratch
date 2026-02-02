import numpy as np
from core.network import SimpleNeuralNetwork

def test_save_dir_load_dir_roundtrip(tmp_path):
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)

    nn = SimpleNeuralNetwork(hidden_size=3, seed=42)
    nn.train(X, y, epochs=200, learning_rate=0.1, log_every=200)

    p1 = nn.forward(X)

    save_path = tmp_path / "xor_v1"
    nn.save_dir(save_path)

    nn2 = SimpleNeuralNetwork.load_dir(save_path)
    p2 = nn2.forward(X)

    assert np.max(np.abs(p1 - p2)) == 0.0
