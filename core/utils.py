import numpy as np

def one_hot(y_idx, num_classes):
    out = np.zeros((y_idx.size, num_classes))
    out[np.arange(y_idx.size), y_idx] = 1
    return out

def batch_iterator(X, y, batch_size):
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end]
