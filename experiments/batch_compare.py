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

def run_case(name, batch_size, epochs=3000, lr=0.1, seed=42, momentum=0.0):
    print(f"\n=== {name} (batch_size={batch_size}, momentum={momentum}) ===")
    nn = SimpleNeuralNetwork(hidden_size=3, seed=seed)

    # Train
    nn.train(
        X, y,
        epochs=epochs,
        learning_rate=lr,
        log_every=500,
        batch_size=batch_size,
        shuffle=True,
        task="binary",
        momentum=momentum
    )

    # Final table
    probs, preds = nn.predict(X, output_activation="sigmoid", threshold=0.5)

    print("\nFinal predictions")
    print("x1 x2 | prob    | pred | true")
    print("-----------------------------")
    for i in range(len(X)):
        x1, x2 = int(X[i, 0]), int(X[i, 1])
        p = float(probs[i, 0])
        pr = int(preds[i, 0])
        tr = int(y[i, 0])
        print(f" {x1}  {x2} | {p:0.6f} |  {pr}   |  {tr}")

def main():
    # Full-batch (GD)
    run_case("Full-batch (GD)", batch_size=4, momentum=0.9)

    # Mini-batch
    run_case("Mini-batch", batch_size=2, momentum=0.9)

    # SGD (batch_size=1)
    run_case("SGD", batch_size=1, lr=0.05, momentum=0.5)



if __name__ == "__main__":
    main()
