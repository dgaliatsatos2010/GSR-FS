from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class GSRSelector(BaseEstimator, TransformerMixin):
    """Geometry-Supported Residual Feature Selection (GSR-FS), prototype v0.12.

    GSR-FS is fully unsupervised. ``y`` is accepted only for scikit-learn API
    compatibility and is never used for feature selection.

    Core components
    ---------------
    1. feature-wise additive pair-geometry profiles;
    2. permutation-calibrated structural support;
    3. Sample-Geometric Coverage (SGC), which downweights geometry concentrated
       on a small set of samples;
    4. non-negative residual novelty relative to already selected feature
       geometry profiles;
    5. max-null permutation stopping in automatic mode.

    v0.4 adds two acceleration engines while retaining reference implementations:

    * ``support_engine='vectorized'`` evaluates permutation nulls in batches and
      computes sample coverage through a sparse pair-incidence operator;
    * ``residual_engine='batched_cd'`` solves all candidate non-negative residual
      projections at a greedy step together using Gram-space coordinate descent.

    The reference engines remain available for numerical equivalence tests.

    v0.5 froze a softer residual exponent ``gamma=0.5`` after a development-only
    sensitivity study. This choice remains frozen before external CC18 evaluation.

    v0.7 does not alter the frozen selection objective. It adds release, CLI,
    provenance, and publication-gate infrastructure around this estimator.

    v0.12 still does not alter the frozen selection objective. Structural-group
    output is implemented in a separate interpretation layer and never changes
    this estimator's selected subset.

    v0.6 separates the random stream used for pair subsampling from the stream
    used for permutation calibration, and uses a shared bank of sample-index
    permutations across features. This removes a Monte-Carlo confound in pair-
    budget studies and makes the stochastic statistic equivariant to a pure
    reordering of feature columns (up to deterministic tie handling).

    Parameters
    ----------
    n_features : int or "auto", default="auto"
        Fixed number of selected features, or automatic stopping.
    n_permutations : int, default=200
        Within-feature permutations used for null calibration.
    alpha : float, default=0.05
        Family-wise max-null calibration level used when n_features="auto".
    max_pairs : int, default=50000
        Maximum unordered observation pairs used when sampling is required.
    pair_sampler : {"balanced", "random"}, default="balanced"
        Approximation used when all n(n-1)/2 pairs do not fit within max_pairs.
        ``balanced`` uses a randomized round-robin edge schedule so sample
        endpoint counts are nearly equal; ``random`` retains the nested random
        pair stream used for legacy sensitivity checks.
    robust : bool, default=True
        If True, use median/MAD scaling followed by clipping before geometry
        construction. If False, use mean/standard-deviation scaling.
    clip_z : float or None, default=4.5
        Symmetric clipping threshold after robust scaling. Set None to disable.
    coverage_power : float, default=2.0
        Exponent applied to SGC in the effective-support statistic.
    use_residual : bool, default=True
        If True, penalize geometry already represented by selected features.
    residual_power : float, default=0.5
        Exponent gamma in ``effective_support * residual_novelty**gamma``.
    allow_empty : bool, default=True
        In automatic mode, allow zero selected features when no candidate exceeds
        the max-null threshold.
    support_engine : {"reference", "vectorized"}, default="reference"
        Permutation-support implementation. Both implement the same statistic.
    permutation_batch_size : int, default=16
        Number of permutations processed together by the vectorized support
        engine. Lower this value to reduce peak memory.
    residual_engine : {"auto", "batched_cd", "nnls"}, default="auto"
        Residual-projection implementation. ``nnls`` is the SciPy reference;
        ``batched_cd`` is the accelerated v0.4 solver.
    residual_max_iter : int, default=1000
        Maximum coordinate-descent sweeps for ``batched_cd``.
    residual_tol : float, default=1e-12
        Maximum coefficient change required for coordinate-descent convergence.
    random_state : int or None, default=None
        Reproducibility seed.
    eps : float, default=1e-12
        Numerical stabilization constant.
    """

    def __init__(
        self,
        n_features="auto",
        n_permutations=200,
        alpha=0.05,
        max_pairs=50000,
        pair_sampler="balanced",
        robust=True,
        clip_z=4.5,
        coverage_power=2.0,
        use_residual=True,
        residual_power=0.5,
        allow_empty=True,
        support_engine="reference",
        permutation_batch_size=16,
        residual_engine="auto",
        residual_max_iter=1000,
        residual_tol=1e-12,
        random_state=None,
        eps=1e-12,
    ):
        self.n_features = n_features
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.max_pairs = max_pairs
        self.pair_sampler = pair_sampler
        self.robust = robust
        self.clip_z = clip_z
        self.coverage_power = coverage_power
        self.use_residual = use_residual
        self.residual_power = residual_power
        self.allow_empty = allow_empty
        self.support_engine = support_engine
        self.permutation_batch_size = permutation_batch_size
        self.residual_engine = residual_engine
        self.residual_max_iter = residual_max_iter
        self.residual_tol = residual_tol
        self.random_state = random_state
        self.eps = eps

    def _coerce_X_and_names(self, X, feature_names=None):
        inferred = None
        if hasattr(X, "columns"):
            inferred = [str(c) for c in X.columns]
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array-like object.")
        if X_arr.shape[0] < 3:
            raise ValueError("GSRSelector requires at least 3 samples.")
        if X_arr.shape[1] < 2:
            raise ValueError("GSRSelector requires at least 2 features.")
        if not np.isfinite(X_arr).all():
            raise ValueError("X contains NaN or infinite values. Impute/clean before fitting.")

        if feature_names is not None:
            names = [str(x) for x in feature_names]
        elif inferred is not None:
            names = inferred
        else:
            names = [f"x{i}" for i in range(X_arr.shape[1])]
        if len(names) != X_arr.shape[1]:
            raise ValueError("feature_names must have one name per input feature.")
        return X_arr, np.asarray(names, dtype=object)

    def _scale_fit(self, X):
        eps = float(self.eps)
        if bool(self.robust):
            center = np.median(X, axis=0)
            mad = np.median(np.abs(X - center), axis=0)
            scale = 1.4826 * mad
            q25, q75 = np.percentile(X, [25, 75], axis=0)
            iqr_scale = (q75 - q25) / 1.349
            sd = X.std(axis=0, ddof=0)
            scale = np.where(scale > eps, scale, iqr_scale)
            scale = np.where(scale > eps, scale, sd)
            scale = np.where(scale > eps, scale, 1.0)
            Z = (X - center) / scale
            if self.clip_z is not None:
                c = float(self.clip_z)
                if c <= 0:
                    raise ValueError("clip_z must be positive or None.")
                Z = np.clip(Z, -c, c)
        else:
            center = X.mean(axis=0)
            scale = X.std(axis=0, ddof=0)
            scale = np.where(scale > eps, scale, 1.0)
            Z = (X - center) / scale

        self.location_ = center
        self.scale_ = scale
        return Z

    def _sample_pairs_random(self, n, rng, target):
        # Fixed-size random stream + insertion-ordered de-duplication. For the
        # same seed and n, smaller budgets are exact prefixes of larger ones.
        pairs = {}
        batch = 8192
        while len(pairs) < target:
            a = rng.integers(0, n, size=batch)
            b = rng.integers(0, n, size=batch)
            mask = a != b
            lo = np.minimum(a[mask], b[mask])
            hi = np.maximum(a[mask], b[mask])
            for pair in zip(lo.tolist(), hi.tolist()):
                if pair not in pairs:
                    pairs[pair] = None
                    if len(pairs) >= target:
                        break
        arr = np.asarray(list(pairs.keys()), dtype=int)
        return arr[:, 0], arr[:, 1]

    def _sample_pairs_balanced(self, n, rng, target):
        """Nested randomized round-robin sample of unique unordered pairs.

        The complete graph can be decomposed into matchings (a round-robin
        schedule). Taking a prefix of those matchings gives nearly equal endpoint
        participation for every observation, while a randomized vertex ordering
        prevents the original row order from defining the sampled geometry.
        """
        real = rng.permutation(np.arange(n, dtype=int)).tolist()
        dummy = -1
        if n % 2:
            players = real + [dummy]
        else:
            players = real
        m = len(players)
        fixed = players[0]
        rotating = players[1:]
        out_i = []
        out_j = []
        # m-1 rounds enumerate every unordered pair once (dummy edges skipped).
        for _ in range(m - 1):
            current = [fixed] + rotating
            half = m // 2
            for q in range(half):
                a = current[q]
                b = current[m - 1 - q]
                if a == dummy or b == dummy:
                    continue
                if a < b:
                    out_i.append(a); out_j.append(b)
                else:
                    out_i.append(b); out_j.append(a)
                if len(out_i) >= target:
                    return np.asarray(out_i, dtype=int), np.asarray(out_j, dtype=int)
            # Circle-method rotation with first participant fixed.
            rotating = [rotating[-1]] + rotating[:-1]
        return np.asarray(out_i, dtype=int), np.asarray(out_j, dtype=int)

    def _sample_pairs(self, n, rng):
        total = n * (n - 1) // 2
        if total <= int(self.max_pairs):
            return np.triu_indices(n, 1)
        target = int(self.max_pairs)
        if self.pair_sampler == "balanced":
            return self._sample_pairs_balanced(n, rng, target)
        if self.pair_sampler == "random":
            return self._sample_pairs_random(n, rng, target)
        raise ValueError("pair_sampler must be 'balanced' or 'random'.")

    @staticmethod
    def _geometry_profiles(Xs, pair_i, pair_j):
        d = Xs[pair_i, :] - Xs[pair_j, :]
        return d * d

    def _coverage_from_pair_mass(self, pair_mass, pair_i, pair_j, n):
        sample_mass = (
            np.bincount(pair_i, weights=pair_mass, minlength=n)
            + np.bincount(pair_j, weights=pair_mass, minlength=n)
        )
        sample_mass /= sample_mass.sum() + float(self.eps)
        entropy = -np.sum(sample_mass * np.log(sample_mass + float(self.eps)))
        return float(np.exp(entropy) / n)

    @staticmethod
    def _pair_incidence(pair_i, pair_j, n):
        m = len(pair_i)
        rows = np.repeat(np.arange(m), 2)
        cols = np.column_stack([pair_i, pair_j]).ravel()
        data = np.ones(2 * m, dtype=float)
        return sparse.csr_matrix((data, (rows, cols)), shape=(m, n))

    def _make_permutation_indices(self, n, rng):
        """Create a common Monte-Carlo permutation bank for all features.

        Reusing the same row-index permutations across feature-wise null tests
        is a common-random-numbers design: every feature is evaluated against
        exactly the same Monte-Carlo perturbations. The bank depends only on
        n, B, and the permutation RNG stream, not on feature order or pair budget.
        """
        B = int(self.n_permutations)
        dtype = np.int32 if int(n) < np.iinfo(np.int32).max else np.int64
        out = np.empty((B, int(n)), dtype=dtype)
        base = np.arange(int(n), dtype=dtype)
        for b in range(B):
            out[b] = rng.permutation(base)
        return out

    def _support_scores_reference(self, Xs, G, pair_i, pair_j, permutation_indices):
        n, p = Xs.shape
        B = int(self.n_permutations)
        eps = float(self.eps)
        cp = float(self.coverage_power)

        P = G / (G.sum(axis=0, keepdims=True) + eps)
        total_profile = P.sum(axis=1)

        robust_z = np.zeros(p)
        affinity = np.zeros(p)
        null_median = np.zeros(p)
        null_scale = np.zeros(p)
        sample_coverage = np.zeros(p)
        null_effective = np.zeros((p, B), dtype=float)

        for j in range(p):
            rest = (total_profile - P[:, j]) / max(p - 1, 1)
            rest = rest / (rest.sum() + eps)
            pj = P[:, j]
            observed = float(np.sqrt(np.maximum(pj, 0) * np.maximum(rest, 0)).sum())
            observed_cov = self._coverage_from_pair_mass(G[:, j], pair_i, pair_j, n)

            null_aff = np.empty(B, dtype=float)
            null_cov = np.empty(B, dtype=float)
            xj = Xs[:, j]
            for b in range(B):
                xp = xj[permutation_indices[b]]
                gp = (xp[pair_i] - xp[pair_j]) ** 2
                pp = gp / (gp.sum() + eps)
                null_aff[b] = np.sqrt(
                    np.maximum(pp, 0) * np.maximum(rest, 0)
                ).sum()
                null_cov[b] = self._coverage_from_pair_mass(gp, pair_i, pair_j, n)

            med = float(np.median(null_aff))
            mad = float(np.median(np.abs(null_aff - med)))
            rscale = 1.4826 * mad
            if rscale <= eps:
                rscale = float(np.std(null_aff, ddof=1)) if B > 1 else 0.0
            rscale = max(rscale, eps)

            z_obs = max(0.0, (observed - med) / rscale)
            z_null = np.maximum(0.0, (null_aff - med) / rscale)

            affinity[j] = observed
            null_median[j] = med
            null_scale[j] = rscale
            robust_z[j] = z_obs
            sample_coverage[j] = observed_cov
            null_effective[j, :] = z_null * np.power(null_cov, cp)

        effective_support = robust_z * np.power(sample_coverage, cp)
        return (
            robust_z,
            sample_coverage,
            effective_support,
            affinity,
            null_median,
            null_scale,
            null_effective,
        )

    def _support_scores_vectorized(self, Xs, G, pair_i, pair_j, permutation_indices):
        n, p = Xs.shape
        B = int(self.n_permutations)
        eps = float(self.eps)
        cp = float(self.coverage_power)
        batch_size = int(self.permutation_batch_size)
        incidence = self._pair_incidence(pair_i, pair_j, n)

        P = G / (G.sum(axis=0, keepdims=True) + eps)
        total_profile = P.sum(axis=1)

        robust_z = np.zeros(p)
        affinity = np.zeros(p)
        null_median = np.zeros(p)
        null_scale = np.zeros(p)
        sample_coverage = np.zeros(p)
        null_effective = np.zeros((p, B), dtype=float)

        for j in range(p):
            rest = (total_profile - P[:, j]) / max(p - 1, 1)
            rest = rest / (rest.sum() + eps)
            pj = P[:, j]
            observed = float(np.sqrt(np.maximum(pj, 0) * np.maximum(rest, 0)).sum())
            observed_cov = self._coverage_from_pair_mass(G[:, j], pair_i, pair_j, n)

            null_aff = np.empty(B, dtype=float)
            null_cov = np.empty(B, dtype=float)
            xj = Xs[:, j]

            for start in range(0, B, batch_size):
                stop = min(B, start + batch_size)
                idx = permutation_indices[start:stop]
                XP = xj[idx]
                GP = (XP[:, pair_i] - XP[:, pair_j]) ** 2
                denom = GP.sum(axis=1, keepdims=True) + eps
                PP = GP / denom
                null_aff[start:stop] = np.sqrt(
                    np.maximum(PP, 0.0) * np.maximum(rest[None, :], 0.0)
                ).sum(axis=1)

                # Pair endpoint mass for all permutations in the batch.
                sample_mass = incidence.T.dot(GP.T).T
                sample_mass /= sample_mass.sum(axis=1, keepdims=True) + eps
                entropy = -np.sum(
                    sample_mass * np.log(sample_mass + eps), axis=1
                )
                null_cov[start:stop] = np.exp(entropy) / n

            med = float(np.median(null_aff))
            mad = float(np.median(np.abs(null_aff - med)))
            rscale = 1.4826 * mad
            if rscale <= eps:
                rscale = float(np.std(null_aff, ddof=1)) if B > 1 else 0.0
            rscale = max(rscale, eps)

            z_obs = max(0.0, (observed - med) / rscale)
            z_null = np.maximum(0.0, (null_aff - med) / rscale)

            affinity[j] = observed
            null_median[j] = med
            null_scale[j] = rscale
            robust_z[j] = z_obs
            sample_coverage[j] = observed_cov
            null_effective[j, :] = z_null * np.power(null_cov, cp)

        effective_support = robust_z * np.power(sample_coverage, cp)
        return (
            robust_z,
            sample_coverage,
            effective_support,
            affinity,
            null_median,
            null_scale,
            null_effective,
        )

    def _support_scores(self, Xs, G, pair_i, pair_j, permutation_indices):
        if self.support_engine == "reference":
            return self._support_scores_reference(Xs, G, pair_i, pair_j, permutation_indices)
        if self.support_engine == "vectorized":
            return self._support_scores_vectorized(Xs, G, pair_i, pair_j, permutation_indices)
        raise ValueError("support_engine must be 'vectorized' or 'reference'.")

    def _residuals_nnls(self, H, selected, remaining):
        if not selected:
            return np.ones(len(remaining), dtype=float)
        A = H[:, selected]
        out = np.empty(len(remaining), dtype=float)
        eps = float(self.eps)
        for q, j in enumerate(remaining):
            _, residual_norm = nnls(A, H[:, j])
            denom = float(H[:, j] @ H[:, j]) + eps
            out[q] = np.clip((residual_norm ** 2) / denom, 0.0, 1.0)
        return out

    def _residuals_batched_cd(self, H, selected, remaining):
        if not selected:
            return np.ones(len(remaining), dtype=float)

        S = np.asarray(selected, dtype=int)
        R = np.asarray(remaining, dtype=int)
        A = H[:, S]
        Gss = A.T @ A
        C = A.T @ H[:, R]
        k, r = C.shape
        coef = np.zeros((k, r), dtype=float)
        diag = np.maximum(np.diag(Gss), float(self.eps))
        tol = float(self.residual_tol)
        max_iter = int(self.residual_max_iter)

        for _ in range(max_iter):
            max_change = 0.0
            for i in range(k):
                # Gss[i] @ coef contains the current self-contribution; remove it
                # to perform an exact coordinate minimization subject to a_i >= 0.
                cross = Gss[i, :] @ coef - Gss[i, i] * coef[i, :]
                new_i = np.maximum(0.0, (C[i, :] - cross) / diag[i])
                if r:
                    max_change = max(max_change, float(np.max(np.abs(new_i - coef[i, :]))))
                coef[i, :] = new_i
            if max_change <= tol:
                break

        GC = Gss @ coef
        # H columns are unit norm up to epsilon; use the measured norms for safety.
        target_norm2 = np.sum(H[:, R] * H[:, R], axis=0)
        residual2 = target_norm2 - 2.0 * np.sum(coef * C, axis=0) + np.sum(coef * GC, axis=0)
        residual2 = np.maximum(residual2, 0.0)
        residual = residual2 / (target_norm2 + float(self.eps))
        return np.clip(residual, 0.0, 1.0)

    def _resolved_residual_engine(self, H):
        if self.residual_engine in {"nnls", "batched_cd"}:
            return self.residual_engine
        if self.residual_engine == "auto":
            # v0.6 empirical engineering rule. NNLS is efficient for small
            # geometry matrices, but its repeated tall least-squares solves show
            # a sharp runtime cliff once the pair dimension becomes large. The
            # batched Gram-space solver is numerically equivalent to tight
            # tolerance and is preferred for either wide or tall problems.
            return "batched_cd" if (H.shape[1] >= 50 or H.shape[0] >= 15000) else "nnls"
        raise ValueError("residual_engine must be 'auto', 'batched_cd', or 'nnls'.")

    def _candidate_residuals(self, H, selected, remaining):
        if not bool(self.use_residual):
            return np.ones(len(remaining), dtype=float)
        engine = self._resolved_residual_engine(H)
        if engine == "nnls":
            return self._residuals_nnls(H, selected, remaining)
        return self._residuals_batched_cd(H, selected, remaining)

    def fit(self, X, y=None, feature_names=None):
        X, names = self._coerce_X_and_names(X, feature_names)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = names

        if self.n_features != "auto":
            k = int(self.n_features)
            if not 1 <= k <= p:
                raise ValueError("n_features must be 'auto' or an integer in [1, p].")
        if int(self.n_permutations) < 5:
            raise ValueError("n_permutations must be at least 5.")
        if self.n_features == "auto" and int(self.n_permutations) < 20:
            raise ValueError("automatic mode requires at least 20 permutations.")
        if int(self.max_pairs) < 10:
            raise ValueError("max_pairs must be at least 10.")
        if self.pair_sampler not in {"balanced", "random"}:
            raise ValueError("pair_sampler must be 'balanced' or 'random'.")
        if not 0 < float(self.alpha) < 1:
            raise ValueError("alpha must lie in (0, 1).")
        if float(self.coverage_power) < 0:
            raise ValueError("coverage_power must be non-negative.")
        if float(self.residual_power) < 0:
            raise ValueError("residual_power must be non-negative.")
        if int(self.permutation_batch_size) < 1:
            raise ValueError("permutation_batch_size must be at least 1.")
        if int(self.residual_max_iter) < 1:
            raise ValueError("residual_max_iter must be at least 1.")
        if float(self.residual_tol) <= 0:
            raise ValueError("residual_tol must be positive.")
        if self.support_engine not in {"vectorized", "reference"}:
            raise ValueError("support_engine must be 'vectorized' or 'reference'.")
        if self.residual_engine not in {"auto", "batched_cd", "nnls"}:
            raise ValueError("residual_engine must be 'auto', 'batched_cd', or 'nnls'.")

        # Split pair-sampling and permutation Monte-Carlo randomness so changing
        # max_pairs cannot silently change the null randomization stream.
        seed_seq = np.random.SeedSequence(self.random_state)
        pair_seed, perm_seed = seed_seq.spawn(2)
        pair_rng = np.random.default_rng(pair_seed)
        perm_rng = np.random.default_rng(perm_seed)

        Xs = self._scale_fit(X)
        pair_i, pair_j = self._sample_pairs(n, pair_rng)
        G = self._geometry_profiles(Xs, pair_i, pair_j)
        permutation_indices = self._make_permutation_indices(n, perm_rng)

        (
            support,
            sample_coverage,
            effective_support,
            affinity,
            null_med,
            null_scale,
            null_effective,
        ) = self._support_scores(
            Xs, G, pair_i, pair_j, permutation_indices
        )

        max_null = np.max(null_effective, axis=0)
        try:
            auto_threshold = float(
                np.quantile(max_null, 1.0 - float(self.alpha), method="higher")
            )
        except TypeError:
            auto_threshold = float(
                np.quantile(max_null, 1.0 - float(self.alpha), interpolation="higher")
            )

        adjusted_p = np.asarray(
            [
                (1.0 + np.sum(max_null >= s)) / (len(max_null) + 1.0)
                for s in effective_support
            ],
            dtype=float,
        )

        H = G / (np.linalg.norm(G, axis=0, keepdims=True) + float(self.eps))

        selected = []
        selection_scores = np.zeros(p)
        residual_at_selection = np.full(p, np.nan)
        step_adjusted_p = np.full(p, np.nan)
        path = []
        max_steps = p if self.n_features == "auto" else int(self.n_features)

        for step in range(max_steps):
            remaining = [j for j in range(p) if j not in selected]
            if not remaining:
                break
            residuals = self._candidate_residuals(H, selected, remaining)
            residual_factor = (
                np.ones_like(residuals)
                if not bool(self.use_residual)
                else np.power(residuals, float(self.residual_power))
            )
            scores = effective_support[np.asarray(remaining)] * residual_factor
            q = int(np.argmax(scores))
            best_j = int(remaining[q])
            best_score = float(scores[q])
            best_residual = float(residuals[q])

            if self.n_features == "auto" and best_score < auto_threshold:
                break

            p_step = float(
                (1.0 + np.sum(max_null >= best_score)) / (len(max_null) + 1.0)
            )
            selected.append(best_j)
            selection_scores[best_j] = best_score
            residual_at_selection[best_j] = best_residual
            step_adjusted_p[best_j] = p_step
            path.append(
                {
                    "step": step + 1,
                    "feature_index": best_j,
                    "feature_name": str(names[best_j]),
                    "support_z": float(support[best_j]),
                    "sample_coverage": float(sample_coverage[best_j]),
                    "effective_support": float(effective_support[best_j]),
                    "max_null_adjusted_p": float(adjusted_p[best_j]),
                    "residual_novelty": best_residual,
                    "selection_score": best_score,
                    "conservative_step_p": p_step,
                    "auto_threshold": auto_threshold,
                }
            )

        if self.n_features == "auto" and not selected and not bool(self.allow_empty):
            best_j = int(np.argmax(effective_support))
            selected = [best_j]
            selection_scores[best_j] = effective_support[best_j]
            residual_at_selection[best_j] = 1.0
            step_adjusted_p[best_j] = adjusted_p[best_j]
            path.append(
                {
                    "step": 1,
                    "feature_index": best_j,
                    "feature_name": str(names[best_j]),
                    "support_z": float(support[best_j]),
                    "sample_coverage": float(sample_coverage[best_j]),
                    "effective_support": float(effective_support[best_j]),
                    "max_null_adjusted_p": float(adjusted_p[best_j]),
                    "residual_novelty": 1.0,
                    "selection_score": float(effective_support[best_j]),
                    "conservative_step_p": float(adjusted_p[best_j]),
                    "auto_threshold": auto_threshold,
                    "forced_selection": True,
                }
            )

        self.support_scores_ = support
        self.sample_coverage_ = sample_coverage
        self.effective_support_ = effective_support
        self.affinity_ = affinity
        self.null_affinity_median_ = null_med
        self.null_affinity_scale_ = null_scale
        self.null_effective_support_ = null_effective
        self.max_null_distribution_ = max_null
        self.auto_threshold_ = auto_threshold
        self.adjusted_p_values_ = adjusted_p
        self.residual_novelty_ = residual_at_selection
        self.step_adjusted_p_values_ = step_adjusted_p
        self.feature_scores_ = selection_scores
        self.selected_indices_ = np.asarray(selected, dtype=int)
        self.n_features_selected_ = len(selected)
        self.selection_path_ = path
        self.n_pairs_used_ = len(pair_i)
        self.support_engine_used_ = self.support_engine
        self.residual_engine_used_ = self._resolved_residual_engine(H) if bool(self.use_residual) else "disabled"
        self.rng_policy_ = "split_pair_permutation_streams_v06"
        self.pair_sampling_scheme_ = (
            "balanced_round_robin_stream_v06" if self.pair_sampler == "balanced"
            else "nested_random_insertion_order_stream_v06"
        )
        self.permutation_scheme_ = "common_sample_index_bank_v06"
        # A compact deterministic fingerprint supports sensitivity-study audits
        # without retaining the potentially large B x n permutation matrix.
        import hashlib
        self.permutation_bank_sha256_ = hashlib.sha256(
            np.ascontiguousarray(permutation_indices).view(np.uint8)
        ).hexdigest()

        self.support_ = np.zeros(p, dtype=bool)
        if selected:
            self.support_[self.selected_indices_] = True

        remaining = [j for j in np.argsort(-effective_support) if j not in selected]
        ordered = selected + remaining
        self.ranking_ = np.empty(p, dtype=int)
        for rank, j in enumerate(ordered, start=1):
            self.ranking_[j] = rank
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        X_arr = np.asarray(X)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than fitted data.")
        return X_arr[:, self.support_]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y=y, **fit_params).transform(X)

    def get_support(self, indices=False):
        check_is_fitted(self, "support_")
        return self.selected_indices_.copy() if indices else self.support_.copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_")
        return self.feature_names_in_[self.support_].copy()

    def get_feature_report(self):
        """Return a pandas DataFrame with fitted feature-level diagnostics."""
        check_is_fitted(self, "support_")
        import pandas as pd

        return pd.DataFrame({
            "feature": self.feature_names_in_,
            "selected": self.support_,
            "rank": self.ranking_,
            "support_z": self.support_scores_,
            "sample_coverage": self.sample_coverage_,
            "effective_support": self.effective_support_,
            "max_null_adjusted_p": self.adjusted_p_values_,
            "residual_novelty_at_selection": self.residual_novelty_,
            "selection_score": self.feature_scores_,
            "conservative_step_p": self.step_adjusted_p_values_,
        }).sort_values(["selected", "rank"], ascending=[False, True]).reset_index(drop=True)
