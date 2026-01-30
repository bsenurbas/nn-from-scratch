import numpy as np

def accuracy_binary(y_true, y_pred, threshold=0.5):
    y_hat = (y_pred >= threshold).astype(int)
    return float(np.mean(y_hat == y_true))

def accuracy_multiclass(y_true_onehot, y_pred_probs):
    y_hat = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_true_onehot, axis=1)
    return float(np.mean(y_hat == y_true))
