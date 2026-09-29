from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import nnls
from sklearn.utils.validation import check_is_fitted

from .selector import GSRSelector


def _residual_fraction(target, basis, eps=1e-12):
    """Squared residual fraction for non-negative reconstruction of target.

    Parameters
    ----------
    target : array, shape (n_pairs,)
        Unit-norm geometry profile.
    basis : array, shape (n_pairs, q)
        Unit-norm basis geometry profiles. Empty basis returns 1.
    """
    target = np.asarray(target, dtype=float)
    basis = np.asarray(basis, dtype=float)
    denom = float(target @ target) + float(eps)
    if basis.ndim != 2 or basis.shape[1] == 0:
        return 1.0
    _, resnorm = nnls(basis, target)
    return float(np.clip((resnorm * resnorm) / denom, 0.0, 1.0))


def contextual_geometry_substitution(H, selected, representative, candidate, eps=1e-12):
    """Contextual Geometry Substitution (CGS) similarity in [0, 1].

    The score asks two complementary questions relative to a selected geometry
    basis S:

    1. Is the candidate already representable by S?
    2. If the representative r is replaced by candidate j, can the new basis
       still reconstruct the representative's geometry profile?

    Let R(a|B) be the non-negative squared residual fraction of geometry profile
    a on basis B. Then

        CGS(r,j | S) = 1 - max(R(j | S), R(r | S\\{r} U {j})).

    A value near 1 means j behaves as a contextual substitute for r, rather than
    merely being marginally correlated with it.
    """
    H = np.asarray(H, dtype=float)
    selected = [int(x) for x in selected]
    r = int(representative)
    j = int(candidate)
    if r == j:
        return 1.0
    if r not in selected:
        raise ValueError("representative must belong to selected.")

    basis_s = H[:, selected]
    r_cand = _residual_fraction(H[:, j], basis_s, eps=eps)

    replacement = [x for x in selected if x != r] + [j]
    basis_replaced = H[:, replacement]
    r_rep = _residual_fraction(H[:, r], basis_replaced, eps=eps)

    return float(np.clip(1.0 - max(r_cand, r_rep), 0.0, 1.0))


class GSRSubstitutionSelector(GSRSelector):
    """Frozen GSR selection plus contextual geometry-substitution groups.

    This experimental v0.11 estimator deliberately preserves the exact feature
    selection of :class:`GSRSelector`. After the frozen GSR fit is complete, it
    builds *diagnostic* substitution groups around each selected representative.

    Grouping is not used to change the selected subset. It is intended to answer
    whether two different selected columns may represent essentially the same
    geometry-level information, making exact-column stability overly punitive.

    Parameters
    ----------
    substitution_threshold : float, default=0.85
        Minimum CGS similarity for assigning an unselected feature to a selected
        representative's substitution group. This parameter is diagnostic and
        has no effect on the frozen GSR selected subset.

    All remaining parameters are inherited from GSRSelector.
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
        substitution_threshold=0.85,
    ):
        super().__init__(
            n_features=n_features,
            n_permutations=n_permutations,
            alpha=alpha,
            max_pairs=max_pairs,
            pair_sampler=pair_sampler,
            robust=robust,
            clip_z=clip_z,
            coverage_power=coverage_power,
            use_residual=use_residual,
            residual_power=residual_power,
            allow_empty=allow_empty,
            support_engine=support_engine,
            permutation_batch_size=permutation_batch_size,
            residual_engine=residual_engine,
            residual_max_iter=residual_max_iter,
            residual_tol=residual_tol,
            random_state=random_state,
            eps=eps,
        )
        self.substitution_threshold = substitution_threshold

    def _scaled_with_fitted_parameters(self, X):
        Z = (np.asarray(X, dtype=float) - self.location_) / self.scale_
        if bool(self.robust) and self.clip_z is not None:
            c = float(self.clip_z)
            Z = np.clip(Z, -c, c)
        return Z

    def _reconstruct_fitted_geometry(self, X):
        # Reproduce the deterministic pair stream used by the frozen base fit.
        seed_seq = np.random.SeedSequence(self.random_state)
        pair_seed, _ = seed_seq.spawn(2)
        pair_rng = np.random.default_rng(pair_seed)
        pair_i, pair_j = self._sample_pairs(np.asarray(X).shape[0], pair_rng)
        Z = self._scaled_with_fitted_parameters(X)
        G = self._geometry_profiles(Z, pair_i, pair_j)
        H = G / (np.linalg.norm(G, axis=0, keepdims=True) + float(self.eps))
        return H

    def fit(self, X, y=None, feature_names=None):
        if not 0.0 <= float(self.substitution_threshold) <= 1.0:
            raise ValueError("substitution_threshold must lie in [0, 1].")
        X_arr, names = self._coerce_X_and_names(X, feature_names=feature_names)
        super().fit(X_arr, y=y, feature_names=names)

        p = X_arr.shape[1]
        selected = self.selected_indices_.tolist()
        H = self._reconstruct_fitted_geometry(X_arr)

        sim = np.full((len(selected), p), np.nan, dtype=float)
        groups = {int(r): [int(r)] for r in selected}
        assigned_to = np.full(p, -1, dtype=int)
        assigned_similarity = np.zeros(p, dtype=float)
        for r in selected:
            assigned_to[r] = r
            assigned_similarity[r] = 1.0

        # Only unselected columns can be assigned to a representative. Selected
        # representatives remain distinct basis elements by construction.
        unselected = [j for j in range(p) if j not in selected]
        for a, r in enumerate(selected):
            for j in range(p):
                if j in selected and j != r:
                    continue
                sim[a, j] = contextual_geometry_substitution(
                    H, selected, r, j, eps=float(self.eps)
                )

        thr = float(self.substitution_threshold)
        for j in unselected:
            vals = sim[:, j]
            if not np.isfinite(vals).any():
                continue
            a = int(np.nanargmax(vals))
            val = float(vals[a])
            if val >= thr:
                r = int(selected[a])
                groups[r].append(int(j))
                assigned_to[j] = r
                assigned_similarity[j] = val

        for r in groups:
            groups[r] = sorted(groups[r])

        self.substitution_similarity_ = sim
        self.substitution_groups_ = groups
        self.substitution_representative_ = assigned_to
        self.substitution_assigned_similarity_ = assigned_similarity
        self.substitution_threshold_ = thr
        self.substitution_grouping_is_diagnostic_ = True
        return self

    def get_substitution_groups(self, names=False):
        check_is_fitted(self, "substitution_groups_")
        if not names:
            return {int(k): list(v) for k, v in self.substitution_groups_.items()}
        out = {}
        for r, members in self.substitution_groups_.items():
            out[str(self.feature_names_in_[r])] = [str(self.feature_names_in_[j]) for j in members]
        return out

    def get_group_report(self):
        check_is_fitted(self, "substitution_groups_")
        import pandas as pd

        rows = []
        for r, members in self.substitution_groups_.items():
            for j in members:
                rows.append({
                    "representative_index": int(r),
                    "representative": str(self.feature_names_in_[r]),
                    "member_index": int(j),
                    "member": str(self.feature_names_in_[j]),
                    "is_selected_representative": bool(j == r),
                    "cgs_similarity": 1.0 if j == r else float(self.substitution_assigned_similarity_[j]),
                })
        return pd.DataFrame(rows).sort_values(
            ["representative_index", "is_selected_representative", "cgs_similarity"],
            ascending=[True, False, False],
        ).reset_index(drop=True)


def reference_group_labels(groups, p):
    """Map feature indices to reference substitution-group labels.

    Unassigned features receive unique singleton labels so they are never treated
    as equivalent merely because neither was assigned to a selected group.
    """
    labels = np.arange(int(p), dtype=int) + 10_000_000
    for gid, (rep, members) in enumerate(sorted(groups.items())):
        for j in members:
            labels[int(j)] = gid
    return labels


def substitution_aware_jaccard(a, b, reference_groups, p):
    """Jaccard similarity after mapping columns to frozen reference groups."""
    labels = reference_group_labels(reference_groups, p)
    A = set(int(labels[int(j)]) for j in a)
    B = set(int(labels[int(j)]) for j in b)
    if not A and not B:
        return 1.0
    return float(len(A & B) / max(len(A | B), 1))
