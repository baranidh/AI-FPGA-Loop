"""
dataset.py - Synthetic 4-class classification benchmark for FPGA inference.

Generates a reproducible 8-feature / 4-class Gaussian-cluster dataset.
The NN is trained in FP32; inference quality on quantized FPGA hardware
is the primary closed-loop feedback signal.
"""

import numpy as np

N_FEATURES = 8
N_CLASSES  = 4


def make_dataset(n_train: int = 400, n_test: int = 100, seed: int = 0):
    """
    Generate a synthetic classification dataset.

    Returns
    -------
    X_train : (n_train, 8)  float64, normalised to ~[-2, 2]
    y_train : (n_train,)    int, class labels 0-3
    X_test  : (n_test,  8)  float64
    y_test  : (n_test,  8)  int
    meta    : dict  {mu, sigma, centers, fp32_baseline}
    """
    rng = np.random.default_rng(seed)

    # Class centres placed far apart so 8-bit inference can distinguish them;
    # 4-bit inference will struggle — that's the closed-loop signal.
    centers = rng.standard_normal((N_CLASSES, N_FEATURES)) * 3.0
    std = 0.9

    def _split(n_per_class):
        Xs, ys = [], []
        for c in range(N_CLASSES):
            Xc = rng.standard_normal((n_per_class, N_FEATURES)) * std + centers[c]
            Xs.append(Xc)
            ys.extend([c] * n_per_class)
        X = np.vstack(Xs)
        y = np.array(ys, dtype=np.int64)
        idx = rng.permutation(len(y))
        return X[idx], y[idx]

    X_tr, y_tr = _split(n_train // N_CLASSES)
    X_te, y_te = _split(n_test  // N_CLASSES)

    # Normalise using training statistics only
    mu    = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0).clip(1e-8)
    X_tr  = (X_tr - mu) / sigma
    X_te  = (X_te - mu) / sigma

    meta = {
        "mu":        mu,
        "sigma":     sigma,
        "centers":   centers,
        "n_classes": N_CLASSES,
        "n_features": N_FEATURES,
    }
    return X_tr, y_tr, X_te, y_te, meta
