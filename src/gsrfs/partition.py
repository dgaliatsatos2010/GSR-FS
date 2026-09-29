from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .selector import GSRSelector


def _average_rank01(values):
    """Average ranks scaled to [0,1], with larger values receiving larger ranks."""
    x = np.asarray(values, dtype=float)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        r = 0.5 * (i + j - 1)
        ranks[order[i:j]] = r
        i = j
    return ranks / max(n - 1, 1)


def persistent_balanced_gap_evidence(x, min_fraction=0.10, windows=(2, 4, 8), eps=1e-12):
    """Label-free persistent balanced-gap evidence for one continuous feature.

    The statistic scans admissible order-statistic gaps. A candidate gap receives
    high evidence only when it is large relative to neighboring spacings at every
    available local window and when it yields a reasonably balanced split.

    It is deliberately *not* Hartigan's dip statistic, Hopkins statistic, a
    clustering objective, or a pseudo-label score. It is an experimental v0.9
    marginal partition-evidence statistic, not yet a novelty claim.
    """
    z = np.sort(np.asarray(x, dtype=float))
    n = z.size
    if n < 12 or not np.all(np.isfinite(z)):
        return 0.0
    gaps = np.diff(z)
    positive = gaps[gaps > eps]
    if positive.size == 0:
        return 0.0
    floor = max(float(np.median(positive)) * 0.25, eps)
    lo = max(1, int(np.ceil(min_fraction * n)))
    hi = min(n - 1, int(np.floor((1.0 - min_fraction) * n)))
    best = 0.0
    for cut in range(lo, hi + 1):
        gi = cut - 1
        gap = float(gaps[gi])
        if gap <= eps:
            continue
        ratios = []
        for w in windows:
            left = gaps[max(0, gi - w):gi]
            right = gaps[gi + 1:min(gaps.size, gi + 1 + w)]
            neigh = np.concatenate([left, right])
            neigh = neigh[neigh > eps]
            if neigh.size < 2:
                continue
            denom = max(float(np.median(neigh)), floor)
            ratios.append(gap / denom)
        if not ratios:
            continue
        persistent_ratio = min(ratios)
        excess = max(0.0, persistent_ratio - 1.0)
        q = cut / n
        balance = 4.0 * q * (1.0 - q)
        score = balance * np.log1p(excess)
        if score > best:
            best = score
    return float(best)


class GSRPartitionSelector(BaseEstimator, TransformerMixin):
    """Experimental GSR-PBGE selector (v0.9 research variant).

    This estimator keeps the frozen v0.8 GSR structural-support calculation and
    adds a rank-fused, fully label-free Persistent Balanced-Gap Evidence (PBGE)
    term before the same residual-novelty greedy selection.

    v0.9 intentionally supports *fixed-k only*. Automatic stopping remains owned
    by the calibrated base GSRSelector until a separate null model for the PBGE
    component is validated.
    """

    def __init__(
        self,
        n_features=5,
        partition_weight=0.35,
        min_partition_fraction=0.10,
        gap_windows=(2, 4, 8),
        n_permutations=40,
        max_pairs=20000,
        pair_sampler="balanced",
        robust=True,
        clip_z=4.5,
        coverage_power=2.0,
        residual_power=0.5,
        support_engine="reference",
        residual_engine="auto",
        random_state=None,
        eps=1e-12,
    ):
        self.n_features = n_features
        self.partition_weight = partition_weight
        self.min_partition_fraction = min_partition_fraction
        self.gap_windows = gap_windows
        self.n_permutations = n_permutations
        self.max_pairs = max_pairs
        self.pair_sampler = pair_sampler
        self.robust = robust
        self.clip_z = clip_z
        self.coverage_power = coverage_power
        self.residual_power = residual_power
        self.support_engine = support_engine
        self.residual_engine = residual_engine
        self.random_state = random_state
        self.eps = eps

    def _base_estimator(self, p):
        return GSRSelector(
            n_features=p,
            n_permutations=self.n_permutations,
            max_pairs=self.max_pairs,
            pair_sampler=self.pair_sampler,
            robust=self.robust,
            clip_z=self.clip_z,
            coverage_power=self.coverage_power,
            use_residual=True,
            residual_power=self.residual_power,
            allow_empty=False,
            support_engine=self.support_engine,
            residual_engine=self.residual_engine,
            random_state=self.random_state,
            eps=self.eps,
        )

    def fit(self, X, y=None, feature_names=None):
        if self.n_features == "auto":
            raise ValueError("GSRPartitionSelector v0.9 supports fixed n_features only.")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D.")
        n, p = X_arr.shape
        k = int(self.n_features)
        if not 1 <= k <= p:
            raise ValueError("n_features must lie in [1,p].")
        eta = float(self.partition_weight)
        if not 0.0 <= eta <= 1.0:
            raise ValueError("partition_weight must lie in [0,1].")
        if not 0.0 < float(self.min_partition_fraction) < 0.5:
            raise ValueError("min_partition_fraction must lie in (0,0.5).")

        base = self._base_estimator(p)
        base.fit(X, y=None, feature_names=feature_names)
        self.base_estimator_ = base
        self.n_features_in_ = p
        self.feature_names_in_ = base.feature_names_in_.copy()

        # Use the exact fitted robust scaling of the frozen base estimator.
        Xs = (X_arr - base.location_) / base.scale_
        if self.robust and self.clip_z is not None:
            Xs = np.clip(Xs, -float(self.clip_z), float(self.clip_z))

        pe = np.asarray([
            persistent_balanced_gap_evidence(
                Xs[:, j],
                min_fraction=float(self.min_partition_fraction),
                windows=tuple(int(w) for w in self.gap_windows),
                eps=float(self.eps),
            )
            for j in range(p)
        ])
        struct_rank = _average_rank01(base.effective_support_)
        part_rank = _average_rank01(pe)
        fused = (1.0 - eta) * struct_rank + eta * part_rank

        # Reconstruct the same pair geometry deterministically for the residual stage.
        seed_seq = np.random.SeedSequence(self.random_state)
        pair_seed, _ = seed_seq.spawn(2)
        pair_rng = np.random.default_rng(pair_seed)
        pair_i, pair_j = base._sample_pairs(n, pair_rng)
        G = base._geometry_profiles(Xs, pair_i, pair_j)
        H = G / (np.linalg.norm(G, axis=0, keepdims=True) + float(self.eps))

        selected = []
        feature_scores = np.zeros(p, dtype=float)
        residual_at_selection = np.full(p, np.nan)
        path = []
        for step in range(k):
            remaining = [j for j in range(p) if j not in selected]
            residuals = base._candidate_residuals(H, selected, remaining)
            scores = fused[np.asarray(remaining)] * np.power(
                residuals, float(self.residual_power)
            )
            q = int(np.argmax(scores))
            j = int(remaining[q])
            selected.append(j)
            feature_scores[j] = float(scores[q])
            residual_at_selection[j] = float(residuals[q])
            path.append({
                "step": step + 1,
                "feature_index": j,
                "feature_name": str(self.feature_names_in_[j]),
                "structural_rank": float(struct_rank[j]),
                "partition_evidence": float(pe[j]),
                "partition_rank": float(part_rank[j]),
                "fused_relevance": float(fused[j]),
                "residual_novelty": float(residuals[q]),
                "selection_score": float(scores[q]),
            })

        self.partition_evidence_ = pe
        self.structural_rank_ = struct_rank
        self.partition_rank_ = part_rank
        self.fused_relevance_ = fused
        self.selected_indices_ = np.asarray(selected, dtype=int)
        self.support_ = np.zeros(p, dtype=bool)
        self.support_[self.selected_indices_] = True
        self.feature_scores_ = feature_scores
        self.residual_novelty_ = residual_at_selection
        self.selection_path_ = path
        self.n_features_selected_ = len(selected)
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        X_arr = np.asarray(X)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than fitted data.")
        return X_arr[:, self.support_]

    def fit_transform(self, X, y=None, feature_names=None):
        return self.fit(X, y=y, feature_names=feature_names).transform(X)

    def get_support(self, indices=False):
        check_is_fitted(self, "support_")
        return self.selected_indices_.copy() if indices else self.support_.copy()

    def get_feature_names_out(self):
        check_is_fitted(self, "support_")
        return self.feature_names_in_[self.support_].copy()
