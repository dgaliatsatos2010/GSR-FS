from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .selector import GSRSelector


def _average_rank01(values):
    x = np.asarray(values, dtype=float)
    n = x.size
    order = np.argsort(x, kind='mergesort')
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


def jackknife_geometry_concordance(
    Xs,
    G,
    pair_i,
    pair_j,
    *,
    n_blocks=5,
    coverage_power=2.0,
    random_state=None,
    eps=1e-12,
):
    """Compute label-free leave-block-out geometry concordance per feature.

    This statistic does not rerun feature selection. Samples are divided into
    disjoint blocks. For each leave-block-out geometry, every feature receives
    an observed structural score based on its Bhattacharyya affinity to the
    aggregate remaining-feature geometry and sample-geometric coverage. Scores
    are converted to within-block ranks. The final concordance is a conservative
    lower-envelope rank: median rank minus robust dispersion.

    The function is experimental and is not claimed to provide formal
    algorithmic-stability guarantees.
    """
    Xs = np.asarray(Xs, dtype=float)
    G = np.asarray(G, dtype=float)
    n, p = Xs.shape
    if n_blocks < 3:
        raise ValueError('n_blocks must be at least 3.')
    if n_blocks >= n:
        raise ValueError('n_blocks must be smaller than n_samples.')
    rng = np.random.default_rng(random_state)
    order = rng.permutation(n)
    blocks = np.array_split(order, int(n_blocks))
    fold_ranks = np.zeros((len(blocks), p), dtype=float)

    for b, block in enumerate(blocks):
        excluded = np.zeros(n, dtype=bool)
        excluded[block] = True
        keep = ~(excluded[pair_i] | excluded[pair_j])
        if keep.sum() < max(10, p):
            raise ValueError('Too few retained pairs in a jackknife block.')
        gi = G[keep]
        pi = pair_i[keep]
        pj = pair_j[keep]
        P = gi / (gi.sum(axis=0, keepdims=True) + eps)
        total_profile = P.sum(axis=1)
        raw = np.zeros(p, dtype=float)
        n_kept_samples = n - len(block)
        # Map original sample ids to dense ids for coverage.
        kept_samples = np.where(~excluded)[0]
        dense = np.full(n, -1, dtype=int)
        dense[kept_samples] = np.arange(n_kept_samples)
        dpi, dpj = dense[pi], dense[pj]

        for j in range(p):
            pj_prof = P[:, j]
            rest = (total_profile - pj_prof) / max(p - 1, 1)
            rest = rest / (rest.sum() + eps)
            affinity = float(np.sqrt(np.maximum(pj_prof, 0.0) * np.maximum(rest, 0.0)).sum())
            pair_mass = gi[:, j]
            sample_mass = (
                np.bincount(dpi, weights=pair_mass, minlength=n_kept_samples)
                + np.bincount(dpj, weights=pair_mass, minlength=n_kept_samples)
            )
            sample_mass /= sample_mass.sum() + eps
            entropy = -np.sum(sample_mass * np.log(sample_mass + eps))
            coverage = float(np.exp(entropy) / n_kept_samples)
            raw[j] = affinity * (coverage ** float(coverage_power))
        fold_ranks[b] = _average_rank01(raw)

    center = np.median(fold_ranks, axis=0)
    mad = np.median(np.abs(fold_ranks - center[None, :]), axis=0)
    lower = np.clip(center - 1.4826 * mad, 0.0, 1.0)
    return lower, center, mad, fold_ranks


class GSRConcordanceSelector(BaseEstimator, TransformerMixin):
    """Experimental v0.10 GSR with Jackknife Geometry Concordance (GSR-JGC).

    The frozen GSR structural statistic is retained. JGC only changes the
    fixed-k relevance ordering by fusing full-sample structural rank with a
    conservative leave-block-out geometry rank. No labels or pseudo-labels are
    used. Automatic stopping is intentionally disabled until a separate null
    calibration is validated for the fused statistic.
    """

    def __init__(
        self,
        n_features=5,
        concordance_weight=0.0,
        tie_tolerance=0.10,
        n_blocks=5,
        n_permutations=40,
        max_pairs=20000,
        pair_sampler='balanced',
        robust=True,
        clip_z=4.5,
        coverage_power=2.0,
        residual_power=0.5,
        support_engine='vectorized',
        residual_engine='auto',
        random_state=None,
        eps=1e-12,
    ):
        self.n_features = n_features
        self.concordance_weight = concordance_weight
        self.tie_tolerance = tie_tolerance
        self.n_blocks = n_blocks
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
            n_permutations=int(self.n_permutations),
            max_pairs=int(self.max_pairs),
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
        if self.n_features == 'auto':
            raise ValueError('GSRConcordanceSelector v0.10 supports fixed n_features only.')
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError('X must be 2D.')
        n, p = X_arr.shape
        k = int(self.n_features)
        if not 1 <= k <= p:
            raise ValueError('n_features must lie in [1,p].')
        w = float(self.concordance_weight)
        if not 0.0 <= w <= 1.0:
            raise ValueError('concordance_weight must lie in [0,1].')
        tie_tol = float(self.tie_tolerance)
        if not 0.0 <= tie_tol < 1.0:
            raise ValueError('tie_tolerance must lie in [0,1).')

        base = self._base_estimator(p)
        base.fit(X_arr, y=None, feature_names=feature_names)
        self.base_estimator_ = base
        self.n_features_in_ = p
        self.feature_names_in_ = base.feature_names_in_.copy()

        Xs = (X_arr - base.location_) / base.scale_
        if self.robust and self.clip_z is not None:
            Xs = np.clip(Xs, -float(self.clip_z), float(self.clip_z))

        seed_seq = np.random.SeedSequence(self.random_state)
        pair_seed, _, concord_seed = seed_seq.spawn(3)
        pair_rng = np.random.default_rng(pair_seed)
        pair_i, pair_j = base._sample_pairs(n, pair_rng)
        G = base._geometry_profiles(Xs, pair_i, pair_j)

        lower, center, mad, fold_ranks = jackknife_geometry_concordance(
            Xs,
            G,
            pair_i,
            pair_j,
            n_blocks=int(self.n_blocks),
            coverage_power=float(self.coverage_power),
            random_state=concord_seed,
            eps=float(self.eps),
        )
        struct_rank = _average_rank01(base.effective_support_)
        fused = (1.0 - w) * struct_rank + w * lower

        H = G / (np.linalg.norm(G, axis=0, keepdims=True) + float(self.eps))
        selected = []
        feature_scores = np.zeros(p, dtype=float)
        residual_at_selection = np.full(p, np.nan)
        path = []
        for step in range(k):
            remaining = [j for j in range(p) if j not in selected]
            residuals = base._candidate_residuals(H, selected, remaining)
            base_scores = base.effective_support_[np.asarray(remaining)] * np.power(residuals, float(self.residual_power))
            fused_scores = fused[np.asarray(remaining)] * np.power(residuals, float(self.residual_power))
            if w > 0.0:
                scores = fused_scores
                q = int(np.argmax(scores))
            else:
                scores = base_scores
                best = float(np.max(base_scores))
                if best <= 0.0 or tie_tol <= 0.0:
                    q = int(np.argmax(base_scores))
                else:
                    eligible = np.where(base_scores >= best * (1.0 - tie_tol))[0]
                    # JGC only resolves near-ties; primary GSR score is unchanged.
                    tie_values = lower[np.asarray(remaining, dtype=int)[eligible]]
                    best_tie = np.max(tie_values)
                    tied = eligible[np.where(tie_values == best_tie)[0]]
                    if tied.size > 1:
                        # deterministic secondary tie-break: higher base score, then lower index.
                        local = base_scores[tied]
                        q = int(tied[np.argmax(local)])
                    else:
                        q = int(tied[0])
            j = int(remaining[q])
            selected.append(j)
            feature_scores[j] = float(scores[q])
            residual_at_selection[j] = float(residuals[q])
            path.append({
                'step': step + 1,
                'feature_index': j,
                'feature_name': str(self.feature_names_in_[j]),
                'structural_rank': float(struct_rank[j]),
                'jgc_lower_rank': float(lower[j]),
                'jgc_center_rank': float(center[j]),
                'jgc_rank_mad': float(mad[j]),
                'fused_relevance': float(fused[j]),
                'tie_tolerance': float(tie_tol),
                'residual_novelty': float(residuals[q]),
                'selection_score': float(scores[q]),
            })

        self.jgc_lower_rank_ = lower
        self.jgc_center_rank_ = center
        self.jgc_rank_mad_ = mad
        self.jgc_fold_ranks_ = fold_ranks
        self.structural_rank_ = struct_rank
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
        check_is_fitted(self, 'support_')
        X_arr = np.asarray(X)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:
            raise ValueError('X has a different number of features than fitted data.')
        return X_arr[:, self.support_]

    def fit_transform(self, X, y=None, feature_names=None):
        return self.fit(X, y=y, feature_names=feature_names).transform(X)

    def get_support(self, indices=False):
        check_is_fitted(self, 'support_')
        return self.selected_indices_.copy() if indices else self.support_.copy()

    def get_feature_names_out(self):
        check_is_fitted(self, 'support_')
        return self.feature_names_in_[self.support_].copy()
