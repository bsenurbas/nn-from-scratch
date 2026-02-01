import numpy as np
from core.network import SimpleNeuralNetwork

def pretty_print(title, X, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    print(f"\n{title}")
    print("x1 x2 | prob    | pred")
    print("----------------------")
    for (x1, x2), p, yhat in zip(X, probs.flatten(), preds.flatten()):
        print(f"{int(x1):>2} {int(x2):>2} | {p:0.6f} |  {int(yhat)}")

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

probs_before = nn.predict_proba(X, task="binary")
pretty_print("Before training", X, probs_before)

nn.train(X, y, epochs=5000, learning_rate=0.1, log_every=500, batch_size=2, shuffle=True)

probs_after = nn.predict_proba(X, task="binary")
pretty_print("After training", X, probs_after)

nn.save("artifacts/xor_weights.npz")

nn2 = SimpleNeuralNetwork(hidden_size=3, seed=42)
nn2.load("artifacts/xor_weights.npz")

probs_loaded = nn2.forward(X)
pretty_print("Loaded model output", X, probs_loaded)

print("\nMax abs diff (after vs loaded):", np.max(np.abs(probs_after - probs_loaded)))

