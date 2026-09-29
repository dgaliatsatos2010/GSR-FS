from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits


@dataclass(frozen=True)
class RealDataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    groups: np.ndarray | None = None
    source: str = ""
    panel_status: str = "development_reused"


def load_builtin_benchmarks(include_digits=True):
    """Load reproducible public datasets bundled with scikit-learn.

    These datasets require no network access, making the benchmark exactly
    reproducible in CI. Class labels are returned only for downstream evaluation.
    """
    specs = [
        ("iris", load_iris),
        ("wine", load_wine),
        ("breast_cancer", load_breast_cancer),
    ]
    if include_digits:
        specs.append(("digits", load_digits))

    out = []
    for name, loader in specs:
        bunch = loader()
        X = np.asarray(bunch.data, dtype=float)
        y = np.asarray(bunch.target)
        names = getattr(bunch, "feature_names", None)
        if names is None:
            names = [f"x{i}" for i in range(X.shape[1])]
        out.append(
            RealDataset(
                name=name,
                X=X,
                y=y,
                feature_names=tuple(map(str, names)),
                source="scikit-learn bundled dataset",
                panel_status="development_reused",
            )
        )
    return out


def default_k_grid(p, n_classes):
    """Small preregisterable fixed-k grid for early real-data experiments."""
    candidates = [
        max(2, int(n_classes)),
        max(5, 2 * int(n_classes)),
        max(10, 4 * int(n_classes)),
        max(2, p // 2),
    ]
    return sorted({min(p, int(k)) for k in candidates if min(p, int(k)) >= min(p, int(n_classes))})


def publication_k_grid(p):
    """Label-independent fixed-k grid for publication-grade benchmarking.

    The grid depends only on the number of predictors, never on class labels.
    Full-dimensional k=p is excluded whenever a genuine reduced subset exists.
    """
    p = int(p)
    if p < 2:
        return [p]
    candidates = [
        2, 5, 10, 20,
        max(2, int(round(0.10 * p))),
        max(2, int(round(0.25 * p))),
        max(2, int(round(0.50 * p))),
    ]
    grid = sorted({min(p, int(k)) for k in candidates if int(k) >= 1})
    reduced = [k for k in grid if k < p]
    return reduced if reduced else [p]



def load_statsmodels_extension_benchmarks():
    """Load three real classification datasets bundled with statsmodels.

    This is a network-free extension panel. It is deliberately separate from
    OpenML-CC18 and must not be described as a replacement for CC18.
    """
    try:
        import statsmodels.datasets as smd
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for the offline extension panel; install gsrfs[benchmark]."
        ) from exc

    out = []

    anes = smd.anes96.load_pandas()
    X = np.asarray(anes.exog, dtype=float)
    y = np.asarray(anes.endog, dtype=int)
    out.append(RealDataset(
        name="anes96",
        X=X,
        y=y,
        feature_names=tuple(map(str, anes.exog_name)),
        source="statsmodels.datasets.anes96",
        panel_status="offline_extension",
    ))

    mode = smd.modechoice.load_pandas()
    X = np.asarray(mode.exog, dtype=float)
    y = np.asarray(mode.endog, dtype=int)
    groups = np.asarray(mode.data["individual"], dtype=int)
    out.append(RealDataset(
        name="modechoice",
        X=X,
        y=y,
        feature_names=tuple(map(str, mode.exog_name)),
        groups=groups,
        source="statsmodels.datasets.modechoice",
        panel_status="offline_extension",
    ))

    spector = smd.spector.load_pandas()
    X = np.asarray(spector.exog, dtype=float)
    y = np.asarray(spector.endog, dtype=int)
    out.append(RealDataset(
        name="spector",
        X=X,
        y=y,
        feature_names=tuple(map(str, spector.exog_name)),
        source="statsmodels.datasets.spector",
        panel_status="offline_extension",
    ))
    return out


def load_offline_real_panel(include_digits=True):
    """Return the frozen v0.8 offline real-data panel."""
    return load_builtin_benchmarks(include_digits=include_digits) + load_statsmodels_extension_benchmarks()


def downstream_k(p):
    """Frozen v0.8 single-k rule depending only on predictor dimension."""
    import math
    p = int(p)
    if p <= 2:
        return p
    return min(p - 1, max(2, int(math.ceil(math.sqrt(p)))))
