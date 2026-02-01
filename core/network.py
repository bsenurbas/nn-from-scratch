import numpy as np
from core.activations import sigmoid, sigmoid_derivative, softmax
from core.losses import bce, cross_entropy
from core.metrics import accuracy_binary, accuracy_multiclass
from core.utils import batch_iterator

class SimpleNeuralNetwork:
    def __init__(self, input_size=2, hidden_size=3, output_size=1, seed=42):
        np.random.seed(seed)

        # weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
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


    def predict(self, X, output_activation="sigmoid", threshold=0.5):
        probs = self.forward(X, output_activation=output_activation)

        if output_activation == "sigmoid":
            preds = (probs >= threshold).astype(int)
        elif output_activation == "softmax":
            preds = np.argmax(probs, axis=1)
        else:
            raise ValueError("output_activation must be 'sigmoid' or 'softmax'")

        return probs, preds


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
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path: str):
        data = np.load(path)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]

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
