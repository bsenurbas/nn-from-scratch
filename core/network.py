import numpy as np
import json
from pathlib import Path
from core.activations import sigmoid, sigmoid_derivative, softmax
from core.losses import bce, cross_entropy
from core.metrics import accuracy_binary, accuracy_multiclass
from core.utils import batch_iterator, xavier_uniform

class SimpleNeuralNetwork:
    """
    Public API (stable):
      - fit(X, y, *, epochs, learning_rate, batch_size, shuffle, task, log_every, momentum) -> self
      - predict_proba(X, task=...) -> np.ndarray
      - predict(X, task=...) -> np.ndarray
      - score(X, y, task=...) -> float
      - evaluate(X, y, task=...) -> dict: {"loss": float, "acc": float}
      - save_dir(path) / load_dir(path)
      - save(path) / load(path)  # backward-compatible wrappers

    Conventions:
      - task="binary": y can be shape (n,) or (n,1)
      - task="multiclass": y can be class ids (n,) or one-hot (n,C)
    """
    def __init__(self, input_size=2, hidden_size=3, output_size=1, seed=42):
        np.random.seed(seed)

        # weights and biases
        self.W1 = xavier_uniform(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = xavier_uniform(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    
        # Momentum velocities (initially zero)
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)

        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)


    def forward(self, X, output_activation="sigmoid"):
        self.X = X

        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2

        if output_activation == "sigmoid":
            self.y_pred = sigmoid(self.z2)
        elif output_activation == "softmax":
            self.y_pred = softmax(self.z2)
        else:
            raise ValueError("output_activation must be 'sigmoid' or 'softmax'")

        return self.y_pred


    def predict_proba(self, X, task="binary"):
        if task == "binary":
            return self.forward(X, output_activation="sigmoid")

        elif task == "multiclass":
            return self.forward(X, output_activation="softmax")

        else:
            raise ValueError("task must be 'binary' or 'multiclass'")


    def predict(self, X, task="binary", threshold=0.5):
        probs = self.predict_proba(X, task=task)

        if task == "binary":
            return (probs >= threshold).astype(int)

        elif task == "multiclass":
            return np.argmax(probs, axis=1)

        else:
            raise ValueError("task must be 'binary' or 'multiclass'")


    def evaluate(self, X, y, *, task="binary"):
        """
        Standard evaluation entry point.

        Returns dict:
        - loss: float
        - acc: float
        """
        probs = self.predict_proba(X, task=task)

        if task == "binary":
            # y: (n,1) or (n,)
            y_true = y.reshape(-1, 1) if y.ndim == 1 else y
            loss = float(bce(y_true, probs))
            acc = float(accuracy_binary(y_true, probs))
            return {"loss": loss, "acc": acc}

        if task == "multiclass":
            # y can be class ids (n,) or one-hot (n,C)
            if y.ndim == 1:
                y_onehot = np.eye(probs.shape[1])[y.astype(int)]
            else:
                y_onehot = y

            loss = float(cross_entropy(y_onehot, probs))
            acc = float(accuracy_multiclass(y_onehot, probs))
            return {"loss": loss, "acc": acc}

        raise ValueError(f"Unknown task: {task}")




    def backward(self, y_true, learning_rate=0.1, loss_type="bce", momentum=0.9):
        m = y_true.shape[0]

        # Output layer gradient
        if loss_type in ["bce", "ce"]:
            dL_dz2 = (self.y_pred - y_true) / m
        else:
            raise ValueError("loss_type must be 'bce' or 'ce'")

        dL_dW2 = np.dot(self.a1.T, dL_dz2)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        # Hidden layer
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_da1 * sigmoid_derivative(self.a1)

        dL_dW1 = np.dot(self.X.T, dL_dz1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        # momentum update (classic)
        if momentum and momentum > 0.0:
            # momentum update (classic)
            self.vW2 = momentum * self.vW2 - learning_rate * dL_dW2
            self.vb2 = momentum * self.vb2 - learning_rate * dL_db2
            self.vW1 = momentum * self.vW1 - learning_rate * dL_dW1
            self.vb1 = momentum * self.vb1 - learning_rate * dL_db1

            self.W2 += self.vW2
            self.b2 += self.vb2
            self.W1 += self.vW1
            self.b1 += self.vb1
        else:
            # plain SGD update
            self.W2 -= learning_rate * dL_dW2
            self.b2 -= learning_rate * dL_db2
            self.W1 -= learning_rate * dL_dW1
            self.b1 -= learning_rate * dL_db1

    def save(self, path: str):
        self.save_dir(path)

    def load(self, path: str):
        loaded = self.__class__.load_dir(path)
        self.W1, self.b1 = loaded.W1, loaded.b1
        self.W2, self.b2 = loaded.W2, loaded.b2
        self.vW1, self.vb1 = loaded.vW1, loaded.vb1
        self.vW2, self.vb2 = loaded.vW2, loaded.vb2


    def train(self, X, y, epochs=5000, learning_rate=0.1, log_every=500,
          batch_size=None, shuffle=True, task="binary", momentum=0.0):

        n = X.shape[0]
        if batch_size is None or batch_size <= 0:
            batch_size = n  # full batch

        for epoch in range(1, epochs + 1):
            # Shuffle
            if shuffle:
                idx = np.random.permutation(n)
                X_epoch = X[idx]
                y_epoch = y[idx]
            else:
                X_epoch = X
                y_epoch = y

            epoch_loss_sum = 0.0
            correct_sum = 0
            seen = 0

            # Mini-batches (batch_iterator ile)
            for Xb, yb in batch_iterator(X_epoch, y_epoch, batch_size):
                bs = Xb.shape[0]
 
                if task == "binary":
                    y_pred = self.forward(Xb, output_activation="sigmoid")
                    loss = bce(yb, y_pred)
                    correct_sum += accuracy_binary(yb, y_pred) * bs
                    self.backward(yb, learning_rate=learning_rate, loss_type="bce", momentum=momentum)

                elif task == "multiclass":
                    y_pred = self.forward(Xb, output_activation="softmax")
                    loss = cross_entropy(yb, y_pred)
                    correct_sum += accuracy_multiclass(yb, y_pred) * bs
                    self.backward(yb, learning_rate=learning_rate, loss_type="ce", momentum=momentum)
                else:
                    raise ValueError("task must be 'binary' or 'multiclass'")

                # Batch katkısı
                epoch_loss_sum += loss * bs
                seen += bs

            #Epoch metrics 
            epoch_loss = epoch_loss_sum / seen
            epoch_acc = correct_sum / seen

            if epoch % log_every == 0 or epoch == 1:
                print(f"Epoch {epoch:5d} | Loss: {epoch_loss:.6f} | Acc: {epoch_acc:.2f}")
    
    def fit(
        self,
        X,
        y,
        *,
        epochs=1000,
        learning_rate=0.1,
        batch_size=None,
        shuffle=True,
        task="binary",
        log_every=100,
        momentum=0.0,
    ):
        """
        Scikit-learn style wrapper around train().
        Returns self for chaining.
        """
        self.train(
            X,
            y,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            shuffle=shuffle,
            task=task,
            log_every=log_every,
            momentum=momentum,
        )
        return self

    def score(self, X, y, *, task="binary"):
        """
        Returns accuracy for binary or multiclass tasks.

        - binary: y can be shape (n,1) or (n,)
        - multiclass: y can be class ids (n,) or one-hot (n,C)
        """
        preds = self.predict(X, task=task)

        if task == "binary":
            y_true = y.reshape(-1) if hasattr(y, "reshape") else y
            return float((preds.reshape(-1) == y_true).mean())

        if task == "multiclass":
            if y.ndim == 2:
                y_true = np.argmax(y, axis=1)
            else:
                y_true = y
            return float((preds == y_true).mean())
        
        raise ValueError(f"Unknown task: {task}")

    def get_config(self):
        return {
            "model_type": "SimpleNeuralNetwork",
            "version": 1,
            "input_size": int(self.W1.shape[0]),
            "hidden_size": int(self.W1.shape[1]),
            "output_size": int(self.W2.shape[1]),
            "init": "xavier_uniform",
        }

    def save_dir(self, save_dir: str):
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)

        # Ağırlıkları kaydet
        np.savez(
            d / "weights.npz",
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2
        )

        # Model konfigürasyonu
        config = {
            "input_size": int(self.W1.shape[0]),
            "hidden_size": int(self.W1.shape[1]),
            "output_size": int(self.W2.shape[1]),
        }

        (d / "config.json").write_text(
            json.dumps(config, indent=2),
            encoding="utf-8"
        )


    @classmethod
    def load_dir(cls, load_dir: str):
        d = Path(load_dir)

        config = json.loads(
            (d / "config.json").read_text(encoding="utf-8")
        )

        nn = cls(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            output_size=config["output_size"],
            seed=42
        )

        data = np.load(d / "weights.npz")
        nn.W1 = data["W1"]
        nn.b1 = data["b1"]
        nn.W2 = data["W2"]
        nn.b2 = data["b2"]

        # Momentum buffer’ları sıfırla
        nn.vW1 = np.zeros_like(nn.W1)
        nn.vb1 = np.zeros_like(nn.b1)
        nn.vW2 = np.zeros_like(nn.W2)
        nn.vb2 = np.zeros_like(nn.b2)

        return nn

