import numpy as np
from core.network import SimpleNeuralNetwork

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

nn = SimpleNeuralNetwork(hidden_size=3, seed=42)

print("Before training:")
print(nn.forward(X))

nn.train(X, y, epochs=5000, learning_rate=0.5, log_every=500)

print("\nAfter training:")
print(nn.forward(X))
