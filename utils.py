import numpy as np


def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])


def get_vocabulary_size(X):
    return max([max(x) for x in X]) + 1  # plus the 0th word


def fit_in_vocabulary(X, voc_size):
    return [[w for w in x if w < voc_size] for x in X]


def batch_generator(X, y, batch_size):
    size = X.shape[0]
    X_copy = X.copy()
    y_copy = y.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            continue

# Metrics


def f_macro(y_true, y_pred):
    labels = (0, 2)  # negative and positive classes' indices
    f = 0
    for label in labels:
        tp = (y_true[:, label] + y_pred[:, label] == 2).sum()
        fp = (y_true[:, label] - y_pred[:, label] == -1).sum()
        fn = (y_true[:, label] - y_pred[:, label] == 1).sum()
        precision = float(tp) / (tp + fp) if tp + fp != 0 else 0
        recall = float(tp) / (tp + fn) if tp + fn != 0 else 0
        f_local = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        f += f_local / len(labels)
    return f


def f_micro(y_true, y_pred):
    labels = (0, 2)  # negative and positive classes' indices
    f = 0
    tp, fp, fn = 0, 0, 0
    for label in labels:
        tp += (y_true[:, label] + y_pred[:, label] == 2).sum()
        fp += (y_true[:, label] - y_pred[:, label] == -1).sum()
        fn += (y_true[:, label] - y_pred[:, label] == 1).sum()
    precision = float(tp) / (tp + fp) if tp + fp != 0 else 0
    recall = float(tp) / (tp + fn) if tp + fn != 0 else 0
    f = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return f


if __name__ == "__main__":
    # Test batch generator
    gen = batch_generator(np.array(['a', 'b', 'c', 'd']), np.array([1, 2, 3, 4]), 2)
    for _ in range(8):
        xx, yy = gen.next()
        print xx, yy
