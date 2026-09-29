from __future__ import annotations

from copy import deepcopy
import numpy as np
from sklearn.utils.validation import check_is_fitted

from .equivalence import GSRSubstitutionSelector


class GSRStructuralGroupSelector(GSRSubstitutionSelector):
    """Frozen GSR selection with human-readable structural-group output.

    This v0.12 estimator is an *interpretation layer*, not a new selection
    objective and not a feature-extraction method. It preserves the exact
    selected subset produced by the frozen :class:`GSRSelector` and organizes
    each selected representative together with any CGS contextual substitutes
    identified by :class:`GSRSubstitutionSelector`.

    A structural group therefore has the form

        SG01 = {representative, contextual substitutes}

    while ``transform`` continues to return only the original selected
    representative columns. No averaging, projection, latent component, or
    synthetic feature is created.

    Parameters
    ----------
    substitution_threshold : float, default=0.85
        Frozen v0.11 CGS diagnostic threshold. This affects only group
        membership and never changes the selected representative subset.

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
            substitution_threshold=substitution_threshold,
        )

    def fit(self, X, y=None, feature_names=None):
        super().fit(X, y=y, feature_names=feature_names)

        groups = []
        feature_to_group = np.full(self.n_features_in_, -1, dtype=int)
        feature_to_group_id = np.asarray([None] * self.n_features_in_, dtype=object)

        for pos, r in enumerate(self.selected_indices_.tolist(), start=1):
            group_id = f"SG{pos:02d}"
            members = list(self.substitution_groups_.get(int(r), [int(r)]))
            member_sims = [
                1.0 if int(j) == int(r) else float(self.substitution_assigned_similarity_[int(j)])
                for j in members
            ]

            for j in members:
                feature_to_group[int(j)] = pos - 1
                feature_to_group_id[int(j)] = group_id

            rep_score = float(self.feature_scores_[r]) if hasattr(self, "feature_scores_") else np.nan
            eff = float(self.effective_support_[r]) if hasattr(self, "effective_support_") else np.nan
            cov = float(self.sample_coverage_[r]) if hasattr(self, "sample_coverage_") else np.nan
            resid = float(self.residual_novelty_[r]) if hasattr(self, "residual_novelty_") else np.nan
            adjp = float(self.adjusted_p_values_[r]) if hasattr(self, "adjusted_p_values_") else np.nan

            groups.append({
                "group_id": group_id,
                "selection_step": int(pos),
                "representative_index": int(r),
                "representative": str(self.feature_names_in_[r]),
                "member_indices": [int(j) for j in members],
                "members": [str(self.feature_names_in_[j]) for j in members],
                "n_members": int(len(members)),
                "n_substitutes": int(max(len(members) - 1, 0)),
                "is_singleton": bool(len(members) == 1),
                "representative_selection_score": rep_score,
                "representative_effective_support": eff,
                "representative_sample_coverage": cov,
                "representative_residual_novelty": resid,
                "representative_adjusted_p_value": adjp,
                "mean_member_cgs_similarity": float(np.mean(member_sims)),
                "min_member_cgs_similarity": float(np.min(member_sims)),
                "member_cgs_similarity": {
                    str(self.feature_names_in_[j]): float(s)
                    for j, s in zip(members, member_sims)
                },
                "output_feature": str(self.feature_names_in_[r]),
            })

        self.structural_groups_ = groups
        self.feature_structural_group_index_ = feature_to_group
        self.feature_structural_group_id_ = feature_to_group_id
        self.structural_group_output_is_interpretive_ = True
        self.structural_group_output_creates_latent_features_ = False
        self.structural_group_output_changes_selection_ = False
        return self

    def get_structural_groups(self):
        """Return structural groups as JSON-serializable dictionaries."""
        check_is_fitted(self, "structural_groups_")
        return deepcopy(self.structural_groups_)

    def get_structural_group_report(self):
        """Return one row per structural group as a pandas DataFrame."""
        check_is_fitted(self, "structural_groups_")
        import pandas as pd

        rows = []
        for g in self.structural_groups_:
            rows.append({
                "group_id": g["group_id"],
                "selection_step": g["selection_step"],
                "representative_index": g["representative_index"],
                "representative": g["representative"],
                "n_members": g["n_members"],
                "n_substitutes": g["n_substitutes"],
                "is_singleton": g["is_singleton"],
                "members": " | ".join(g["members"]),
                "representative_selection_score": g["representative_selection_score"],
                "representative_effective_support": g["representative_effective_support"],
                "representative_sample_coverage": g["representative_sample_coverage"],
                "representative_residual_novelty": g["representative_residual_novelty"],
                "representative_adjusted_p_value": g["representative_adjusted_p_value"],
                "mean_member_cgs_similarity": g["mean_member_cgs_similarity"],
                "min_member_cgs_similarity": g["min_member_cgs_similarity"],
                "output_feature": g["output_feature"],
            })
        return pd.DataFrame(rows)

    def get_feature_group_map(self, names=True, include_unassigned=True):
        """Map original features to structural-group identifiers.

        Features not assigned to a selected structural group are returned as
        ``None`` when ``include_unassigned=True`` and omitted otherwise.
        """
        check_is_fitted(self, "structural_groups_")
        out = {}
        for j in range(self.n_features_in_):
            gid = self.feature_structural_group_id_[j]
            if gid is None and not include_unassigned:
                continue
            key = str(self.feature_names_in_[j]) if names else int(j)
            out[key] = None if gid is None else str(gid)
        return out

    def get_representatives(self, names=True):
        """Return the original selected representative columns."""
        check_is_fitted(self, "structural_groups_")
        if names:
            return [str(self.feature_names_in_[j]) for j in self.selected_indices_]
        return self.selected_indices_.copy()

    def transform_representatives(self, X):
        """Alias of transform; no latent group feature is constructed."""
        return self.transform(X)

    def get_group_summary(self):
        """Compact JSON-safe summary of the structural representation."""
        check_is_fitted(self, "structural_groups_")
        n_groups = len(self.structural_groups_)
        n_non_singleton = sum(not g["is_singleton"] for g in self.structural_groups_)
        n_grouped_features = sum(g["n_members"] for g in self.structural_groups_)
        n_substitutes = sum(g["n_substitutes"] for g in self.structural_groups_)
        return {
            "n_structural_groups": int(n_groups),
            "n_non_singleton_groups": int(n_non_singleton),
            "n_grouped_original_features": int(n_grouped_features),
            "n_contextual_substitutes": int(n_substitutes),
            "n_selected_representatives": int(self.n_features_selected_),
            "representative_output_dimension": int(self.n_features_selected_),
            "creates_latent_features": False,
            "changes_frozen_gsr_selection": False,
            "substitution_threshold": float(self.substitution_threshold_),
        }
