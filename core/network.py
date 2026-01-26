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

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def forward(self, X):
        # cache values for backprop
        self.X = X

        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_pred = self.sigmoid(self.z2)

        return self.y_pred

    def backward(self, y_true, learning_rate=0.1):
        """
        Backprop for MSE loss with sigmoid output.
        Shapes (for XOR batch of 4):
        X:     (4,2)
        a1:    (4,3)
        y_pred:(4,1)
        """

        m = y_true.shape[0]  # number of samples in batch

        # dLoss/dy_pred for MSE: d/dy_pred mean((y - y_pred)^2) = 2*(y_pred - y)/m
        dL_dy = (2 * (self.y_pred - y_true)) / m  # (m,1)

        # y_pred = sigmoid(z2)
        dy_dz2 = self.sigmoid_derivative(self.y_pred)  # (m,1)
        dL_dz2 = dL_dy * dy_dz2  # (m,1)

        # z2 = a1.W2 + b2
        dL_dW2 = np.dot(self.a1.T, dL_dz2)  # (3,m)@(m,1) -> (3,1)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)  # (1,1)

        # backprop into hidden layer
        dL_da1 = np.dot(dL_dz2, self.W2.T)  # (m,1)@(1,3)->(m,3)

        # a1 = sigmoid(z1)
        da1_dz1 = self.sigmoid_derivative(self.a1)  # (m,3)
        dL_dz1 = dL_da1 * da1_dz1  # (m,3)

        # z1 = X.W1 + b1
        dL_dW1 = np.dot(self.X.T, dL_dz1)  # (2,m)@(m,3)->(2,3)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)  # (1,3)

        # gradient descent update
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1

    def train(self, X, y, epochs=5000, learning_rate=0.1, log_every=500):
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = self.mse(y, y_pred)
            self.backward(y, learning_rate=learning_rate)

            if epoch % log_every == 0 or epoch == 1:
                print(f"Epoch {epoch:>5} | Loss: {loss:.6f}")
