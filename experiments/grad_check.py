import numpy as np
from core.network import SimpleNeuralNetwork

def numeric_gradient(nn, X, y, param_name, idx, eps=1e-5):
    """
    Central difference:
    dL/dp ≈ (L(p+eps) - L(p-eps)) / (2*eps)
    """
    param = getattr(nn, param_name)
    original = param[idx]

    # p + eps
    param[idx] = original + eps
    y_pred_plus = nn.forward(X)
    loss_plus = nn.bce(y, y_pred_plus)

    # p - eps
    param[idx] = original - eps
    y_pred_minus = nn.forward(X)
    loss_minus = nn.bce(y, y_pred_minus)

    # restore
    param[idx] = original

    return (loss_plus - loss_minus) / (2 * eps)

def analytic_gradient_W1_00(nn, X, y):
    """
    Compute dL/dW1[0,0] analytically using the same math as backward,
    but without updating parameters.
    """
    m = y.shape[0]

    # Forward (fills caches)
    y_pred = nn.forward(X)

    # BCE + sigmoid: dL/dz2 = (y_pred - y) / m
    dL_dz2 = (y_pred - y) / m                    # (m,1)

    # dL/da1 = dL/dz2 * W2^T
    dL_da1 = np.dot(dL_dz2, nn.W2.T)             # (m,3)

    # da1/dz1 = a1*(1-a1)
    dL_dz1 = dL_da1 * nn.sigmoid_derivative(nn.a1)  # (m,3)

    # dL/dW1 = X^T dot dL/dz1
    dL_dW1 = np.dot(nn.X.T, dL_dz1)              # (2,3)

    return dL_dW1[0, 0]

def main():
    # XOR dataset
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)

    nn = SimpleNeuralNetwork(hidden_size=3, seed=42)

    num = numeric_gradient(nn, X, y, "W1", (0,0))
    ana = analytic_gradient_W1_00(nn, X, y)

    abs_diff = abs(num - ana)
    rel_diff = abs_diff / (abs(num) + abs(ana) + 1e-12)

    print("Gradient check for W1[0,0]")
    print(f"Numeric  : {num:.10f}")
    print(f"Analytic : {ana:.10f}")
    print(f"Abs diff : {abs_diff:.10e}")
    print(f"Rel diff : {rel_diff:.10e}")

if __name__ == "__main__":
    main()
