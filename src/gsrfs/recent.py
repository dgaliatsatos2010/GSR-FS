"""Pinned bridge to the public FGMRW-UFS TKDE 2026 author repository.

No FGMRW-UFS source code is vendored.  When explicitly used for benchmarking,
this bridge downloads two pinned upstream files into a temporary/cache directory,
verifies their Git blob SHA-1 values, imports them, and passes X only.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import urllib.request
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

FGMRW_REPOSITORY = "HongtaoGao-code/FGMRW-UFS"
FGMRW_COMMIT = "325a904a28284a879c8a522ca7c4abc56449e9f5"
FGMRW_FILES = {
    "FGMRW-UFS-code.py": {
        "git_blob_sha1": "ff817880c7aa56366e957fe33e751b0db273443d",
        "url": f"https://raw.githubusercontent.com/{FGMRW_REPOSITORY}/{FGMRW_COMMIT}/FGMRW-UFS-code.py",
    },
    "GB.py": {
        "git_blob_sha1": "dd4814cdae5395ccaf35eb2c697e06efdbfb8bfb",
        "url": f"https://raw.githubusercontent.com/{FGMRW_REPOSITORY}/{FGMRW_COMMIT}/GB.py",
    },
}


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def fetch_fgmrw_author_source(cache_dir: str | Path | None = None, *, timeout: float = 30.0) -> Path:
    """Fetch and verify the pinned public author-source snapshot.

    This is benchmark infrastructure only.  It requires network access and does
    not make the upstream code part of the GSR-FS distribution.
    """
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "gsrfs_external" / f"fgmrw_{FGMRW_COMMIT[:12]}"
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, meta in FGMRW_FILES.items():
        path = root / name
        ok = False
        if path.exists():
            try:
                ok = git_blob_sha1(path.read_bytes()) == meta["git_blob_sha1"]
            except OSError:
                ok = False
        if not ok:
            with urllib.request.urlopen(meta["url"], timeout=float(timeout)) as resp:
                payload = resp.read()
            observed = git_blob_sha1(payload)
            if observed != meta["git_blob_sha1"]:
                raise RuntimeError(
                    f"FGMRW source-integrity failure for {name}: expected Git blob "
                    f"{meta['git_blob_sha1']}, observed {observed}."
                )
            path.write_bytes(payload)
    return root


def _load_fgmrw(cache_dir=None):
    root = fetch_fgmrw_author_source(cache_dir)
    # The author file imports `GB` by module name. Load the pinned GB.py first.
    gb_path = root / "GB.py"
    gb_spec = importlib.util.spec_from_file_location("GB", gb_path)
    if gb_spec is None or gb_spec.loader is None:
        raise ImportError("Could not create module spec for pinned FGMRW GB.py")
    gb_mod = importlib.util.module_from_spec(gb_spec)
    sys.modules["GB"] = gb_mod
    gb_spec.loader.exec_module(gb_mod)

    code_path = root / "FGMRW-UFS-code.py"
    spec = importlib.util.spec_from_file_location("gsrfs_fgmrw_author", code_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not create module spec for pinned FGMRW author code")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FGMRWAuthorSelector(BaseEstimator, TransformerMixin):
    """Strict label-blind bridge to pinned FGMRW-UFS author algorithm code.

    Parameters ``alpha=0.1`` and ``s=0.8`` are frozen from the public repository's
    example call; they are *not* tuned using downstream labels.  MinMax scaling is
    fit on X only so continuous variables satisfy the author code's [0,1] numerical
    attribute rule.  Constant columns are placed at the end of the ranking.
    """

    ranking_depends_on_k = False

    def __init__(self, n_features=10, alpha=0.1, s=0.8, cache_dir=None):
        self.n_features = n_features
        self.alpha = alpha
        self.s = s
        self.cache_dir = cache_dir

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        n, p = X.shape
        k = int(self.n_features)
        if not 1 <= k <= p:
            raise ValueError("n_features must lie in [1, p]")
        if not np.isfinite(X).all():
            raise ValueError("FGMRWAuthorSelector requires finite preprocessed X")

        # Strictly label-blind: y is accepted only for sklearn compatibility and ignored.
        var = np.ptp(X, axis=0)
        active = np.flatnonzero(var > 0)
        constants = np.flatnonzero(var <= 0)
        if active.size:
            scaler = MinMaxScaler()
            Xa = scaler.fit_transform(X[:, active])
            # Keep the public implementation's exact attribute-type branch stable.
            Xa = np.clip(Xa, 0.0, 1.0)
            Xa[np.argmin(Xa, axis=0), np.arange(Xa.shape[1])] = 0.0
            Xa[np.argmax(Xa, axis=0), np.arange(Xa.shape[1])] = 1.0
            mod = _load_fgmrw(self.cache_dir)
            rank_active_local = np.asarray(
                mod.FGMRW_UFS(Xa, alpha=float(self.alpha), s=float(self.s)), dtype=int
            ).ravel()
            if rank_active_local.size != active.size or set(rank_active_local.tolist()) != set(range(active.size)):
                raise RuntimeError("FGMRW author code did not return a complete feature permutation")
            ranking = np.concatenate([active[rank_active_local], constants])
        else:
            ranking = np.arange(p, dtype=int)

        self.n_features_in_ = p
        self.ranking_indices_ = ranking.astype(int, copy=True)
        self.selected_indices_ = self.ranking_indices_[:k].copy()
        self.support_ = np.zeros(p, dtype=bool)
        self.support_[self.selected_indices_] = True
        self.external_source_ = (
            f"FGMRW-UFS:TKDE2026:author-source-pinned:{FGMRW_COMMIT}:"
            "strict-X-only:alpha0.1:s0.8"
        )
        self.selection_used_y_ = False
        self.selection_used_ground_truth_class_count_ = False
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices=False):
        check_is_fitted(self, "support_")
        return self.selected_indices_.copy() if indices else self.support_.copy()
