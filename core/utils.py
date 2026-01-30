import numpy as np

def one_hot(y_idx, num_classes):
    out = np.zeros((y_idx.size, num_classes))
    out[np.arange(y_idx.size), y_idx] = 1
    return out

def batch_iterator(X, y, batch_size):
    """
    X ve y verisini mini-batch'lere böler.
    Her iterasyonda (X_batch, y_batch) döndürür.
    """

    n = X.shape[0]

    for start in range(0, n, batch_size):
        end = start + batch_size

        Xb = X[start:end]
        yb = y[start:end]

        yield Xb, yb
