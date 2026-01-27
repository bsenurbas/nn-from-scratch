import numpy as np


class SimpleNeuralNetwork:
    def __init__(self, input_size=2, hidden_size=3, output_size=1, seed=42):
        np.random.seed(seed)

        # weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(sigmoid_output):
        # If a = sigmoid(z), then d(sigmoid)/dz = a * (1 - a)
        return sigmoid_output * (1 - sigmoid_output)

    """@staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)"""
    
    @staticmethod
    def bce(y_true, y_pred, eps=1e-12):
        # log(0) hatasını önlemek için clipping
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def accuracy(y_true, y_pred, threshold=0.5):
        y_hat = (y_pred >= threshold).astype(int)
        return float(np.mean(y_hat == y_true))


    def forward(self, X):
        # cache values for backprop
        self.X = X

        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_pred = self.sigmoid(self.z2)

        return self.y_pred

    def predict(self, X, threshold=0.5):
        probs = self.forward(X)
        preds = (probs >= threshold).astype(int)
        return probs, preds

    
    def backward(self, y_true, learning_rate=0.1):
        """
          BCE loss + sigmoid output için:
         dL/dz2 = (y_pred - y) / m
         """
        m = y_true.shape[0]

        # Output layer gradient (BCE + sigmoid)
        dL_dz2 = (self.y_pred - y_true) / m  # (m,1)

        dL_dW2 = np.dot(self.a1.T, dL_dz2)               # (3,1)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)   # (1,1)

        # Hidden layer
        dL_da1 = np.dot(dL_dz2, self.W2.T)               # (m,3)
        dL_dz1 = dL_da1 * self.sigmoid_derivative(self.a1)

        dL_dW1 = np.dot(self.X.T, dL_dz1)                # (2,3)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)   # (1,3)

        # Update
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

    def train(self, X, y, epochs=5000, learning_rate=0.1, log_every=500, batch_size=None, shuffle=True):
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

            # Mini-batches
            for start in range(0, n, batch_size):
                end = start + batch_size
                Xb = X_epoch[start:end]
                yb = y_epoch[start:end]

                y_pred = self.forward(Xb)
                loss = self.bce(yb, y_pred)

                # Acc on this batch
                y_hat = (y_pred >= 0.5).astype(int)
                correct_sum += int(np.sum(y_hat == yb))

                # Loss weighted by batch size
                bs = Xb.shape[0]
                epoch_loss_sum += float(loss) * bs
                seen += bs

                # Update
                self.backward(yb, learning_rate=learning_rate)

            # Epoch metrics
            epoch_loss = epoch_loss_sum / seen
            epoch_acc = correct_sum / seen

            if epoch % log_every == 0 or epoch == 1:
                print(f"Epoch {epoch:5d} | Loss: {epoch_loss:.6f} | Acc: {epoch_acc:.2f}")

