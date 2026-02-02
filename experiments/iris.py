import numpy as np
from pathlib import Path

from core.network import SimpleNeuralNetwork
from core.utils import one_hot


def load_iris_csv(path: Path):
    # CSV: header + 5 columns (4 features + label)
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    X = data[:, :4].astype(float)
    y = data[:, 4].astype(int)
    return X, y


def standardize_train_test(X_train, X_test):
    # Standardize using train statistics only
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    return X_train, X_test


def train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)

    n_test = int(round(X.shape[0] * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def accuracy(pred, y_true):
    return float(np.mean(pred == y_true))


def main():
    csv_path = Path("data") / "iris.csv"
    X, y = load_iris_csv(csv_path)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

    # Standardize
    X_train, X_test = standardize_train_test(X_train, X_test)

    # One-hot for training
    num_classes = 3
    y_train_oh = one_hot(y_train, num_classes=num_classes)

    # Model
    nn = SimpleNeuralNetwork(
        input_size=4,
        hidden_size=8,
        output_size=3,
        seed=42,
    )

    print("Training Iris multiclass classifier")
    nn.train(
        X_train,
        y_train_oh,
        epochs=2000,
        learning_rate=0.1,
        batch_size=16,
        shuffle=True,
        task="multiclass",
        log_every=200,
        momentum=0.9,
    )

    # Evaluation
    pred_train = nn.predict(X_train, task="multiclass")
    pred_test = nn.predict(X_test, task="multiclass")

    train_acc = accuracy(pred_train, y_train)
    test_acc = accuracy(pred_test, y_test)

    print(f"\nTrain acc: {train_acc:0.3f}")
    print(f"Test  acc: {test_acc:0.3f}")

    # Save / load validation
    model_dir = Path("models") / "iris_v1"
    nn.save_dir(model_dir)

    nn2 = SimpleNeuralNetwork.load_dir(model_dir)
    pred_test_2 = nn2.predict(X_test, task="multiclass")
    test_acc_2 = accuracy(pred_test_2, y_test)

    print(f"Loaded model test acc: {test_acc_2:0.3f}")


if __name__ == "__main__":
    main()
