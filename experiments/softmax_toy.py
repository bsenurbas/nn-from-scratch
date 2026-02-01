import numpy as np
from core.network import SimpleNeuralNetwork
from core.utils import one_hot


def main():
    np.random.seed(42)

    # 3-class 2D toy dataset (3 clusters)
    n_per_class = 100
    K = 3

    c0 = np.random.randn(n_per_class, 2) * 0.6 + np.array([0.0, 0.0])
    c1 = np.random.randn(n_per_class, 2) * 0.6 + np.array([3.0, 0.0])
    c2 = np.random.randn(n_per_class, 2) * 0.6 + np.array([1.5, 2.5])

    X = np.vstack([c0, c1, c2]).astype(float)
    y_idx = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)
    y = one_hot(y_idx, K).astype(float)

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=8, output_size=3, seed=42)

    print("Training softmax classifier on 3-class toy data")
    nn.train(X, y, epochs=3000, learning_rate=0.1, log_every=300, batch_size=32, shuffle=True, task="multiclass")

    # Final accuracy on training set
    probs = nn.forward(X, output_activation="softmax")
    pred = np.argmax(probs, axis=1)
    acc = np.mean(pred == y_idx)
    print(f"\nFinal train acc: {acc:.3f}")

if __name__ == "__main__":
    main()
