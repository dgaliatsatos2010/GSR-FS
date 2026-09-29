from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs, make_moons


def redundant_blocks(seed=0, n=240, noise_features=20):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    if y.size < n:
        y = np.r_[y, 1]
    z1 = rng.normal(np.where(y == 0, -2.0, 2.0), 0.55)
    z2 = rng.normal(np.where(y == 0, -1.4, 1.4), 0.65)
    X = np.column_stack([
        z1,
        z2,
        z1 + rng.normal(0, 0.05, n),
        z2 + rng.normal(0, 0.05, n),
        rng.normal(size=(n, noise_features)),
    ])
    return X, y, [[0, 2], [1, 3]], "redundant_blocks"


def nonlinear_moons(seed=0, n=240, noise_features=20):
    X2, y = make_moons(n_samples=n, noise=0.08, random_state=seed)
    rng = np.random.default_rng(seed + 1000)
    # Monotone/nonlinear redundant versions preserve latent factors imperfectly.
    f0 = X2[:, 0]
    f1 = X2[:, 1]
    X = np.column_stack([
        f0,
        f1,
        np.tanh(f0) + rng.normal(0, 0.02, n),
        np.sign(f1) * np.sqrt(np.abs(f1) + 1e-6) + rng.normal(0, 0.02, n),
        rng.normal(size=(n, noise_features)),
    ])
    return X, y, [[0, 2], [1, 3]], "nonlinear_moons"


def three_factor_blobs(seed=0, n=300, noise_features=30):
    Z, y = make_blobs(
        n_samples=n,
        centers=[[-3, 0, 2], [0, 3, -2], [3, -2, 0]],
        cluster_std=0.7,
        n_features=3,
        random_state=seed,
    )
    rng = np.random.default_rng(seed + 2000)
    X = np.column_stack([
        Z[:, 0], Z[:, 1], Z[:, 2],
        Z[:, 0] + rng.normal(0, 0.08, n),
        Z[:, 1] + rng.normal(0, 0.08, n),
        Z[:, 2] + rng.normal(0, 0.08, n),
        rng.normal(size=(n, noise_features)),
    ])
    return X, y, [[0, 3], [1, 4], [2, 5]], "three_factor_blobs"


def scale_trap(seed=0, n=240, noise_features=20):
    """Signals are modest-scale; nuisance dimensions have huge raw variance."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    s1 = rng.normal(np.where(y == 0, -1.5, 1.5), 0.55)
    s2 = rng.normal(np.where(y == 0, -1.0, 1.0), 0.55)
    nuisance = rng.normal(0, 25.0, size=(n, noise_features))
    X = np.column_stack([s1, s2, nuisance])
    return X, y, [[0], [1]], "scale_trap"


def outlier_contamination(seed=0, n=240, noise_features=20, outlier_fraction=0.08):
    X, y, groups, _ = redundant_blocks(seed, n, noise_features)
    rng = np.random.default_rng(seed + 3000)
    m = max(1, int(round(outlier_fraction * n)))
    idx = rng.choice(n, size=m, replace=False)
    X[idx, 4:] += rng.normal(0, 12.0, size=(m, noise_features))
    return X, y, groups, "outlier_contamination"


def null_independent(seed=0, n=240, p=24):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = rng.integers(0, 2, size=n)
    return X, y, [], "null_independent"


SCENARIOS = {
    "redundant_blocks": redundant_blocks,
    "nonlinear_moons": nonlinear_moons,
    "three_factor_blobs": three_factor_blobs,
    "scale_trap": scale_trap,
    "outlier_contamination": outlier_contamination,
    "null_independent": null_independent,
}
