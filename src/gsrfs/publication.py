from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PublicationGateConfig:
    min_datasets: int = 30
    min_competitors: int = 3
    min_canonical_or_author_methods: int = 2
    max_method_failure_fraction: float = 0.20
    candidate_method: str = "GSR-FS"
    require_source_integrity: bool = True


def _successful_rows(df):
    out = df.copy()
    if "error" in out.columns:
        err = out["error"].fillna("").astype(str).str.strip()
        out = out.loc[err.eq("")]
    if "nmi" in out.columns:
        out = out.loc[np.isfinite(pd.to_numeric(out["nmi"], errors="coerce"))]
    return out


def _source_kind(value):
    text = str(value or "").lower()
    if text.startswith("author_code:") or "author" in text:
        return "author_code"
    if "scikit-feature" in text or text.startswith("canonical:"):
        return "canonical"
    return "native_or_local"


def evaluate_publication_gate(
    results,
    *,
    config=None,
    label_audit=None,
    frozen_hyperparameters=True,
    canonical_source_lock=True,
    source_integrity=None,
):
    """Evaluate whether benchmark evidence clears a conservative claim gate.

    This is an engineering guardrail, not a statistical proof of superiority.
    A passing gate means that minimum evidence/provenance requirements are
    present; manuscript claims must still be supported by the actual statistics.
    """
    config = config or PublicationGateConfig()
    df = results.copy() if isinstance(results, pd.DataFrame) else pd.read_csv(results)
    required = {"dataset", "method"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark results missing required columns: {sorted(missing)}")

    success = _successful_rows(df)
    checks = []

    cand = success.loc[success["method"].astype(str).eq(config.candidate_method)].copy()

    source_integrity_ok = not config.require_source_integrity
    eligible_source_datasets = None
    if source_integrity is not None:
        from .source_integrity import eligible_datasets_from_source_audit
        eligible_source_datasets = eligible_datasets_from_source_audit(source_integrity)
        source_integrity_ok = len(eligible_source_datasets) > 0
        cand = cand.loc[cand["dataset"].astype(str).isin(eligible_source_datasets)]

    checks.append({
        "name": "dataset_source_integrity_audit",
        "passed": bool(source_integrity_ok),
        "observed": (len(eligible_source_datasets) if eligible_source_datasets is not None else 0),
        "required": "audited PASS sources" if config.require_source_integrity else "optional",
    })

    n_datasets = int(cand["dataset"].nunique())
    checks.append({
        "name": "minimum_external_datasets",
        "passed": n_datasets >= config.min_datasets,
        "observed": n_datasets,
        "required": config.min_datasets,
    })

    methods = sorted(set(success["method"].astype(str)) - {config.candidate_method})
    checks.append({
        "name": "minimum_competitors",
        "passed": len(methods) >= config.min_competitors,
        "observed": len(methods),
        "required": config.min_competitors,
    })

    source_col = None
    for c in ["implementation_source", "external_source", "source_kind"]:
        if c in success.columns:
            source_col = c
            break
    canonical_methods = set()
    if source_col is not None:
        for method, sub in success.groupby("method"):
            kinds = {_source_kind(v) if source_col != "source_kind" else str(v) for v in sub[source_col]}
            if kinds & {"canonical", "author_code"}:
                canonical_methods.add(str(method))
    checks.append({
        "name": "canonical_or_author_code_competitors",
        "passed": len(canonical_methods - {config.candidate_method}) >= config.min_canonical_or_author_methods,
        "observed": len(canonical_methods - {config.candidate_method}),
        "required": config.min_canonical_or_author_methods,
    })

    # Failure rate is calculated by unique dataset/method availability, not by k rows.
    fail_ok = True
    worst = 0.0
    if "error" in df.columns:
        pairs = df[["dataset", "method", "error"]].copy()
        pairs["failed"] = pairs["error"].fillna("").astype(str).str.strip().ne("")
        pair_fail = pairs.groupby(["dataset", "method"])["failed"].max().reset_index()
        by_method = pair_fail.groupby("method")["failed"].mean()
        if len(by_method):
            worst = float(by_method.max())
            fail_ok = bool((by_method <= config.max_method_failure_fraction).all())
    checks.append({
        "name": "method_failure_fraction",
        "passed": fail_ok,
        "observed": worst,
        "required": f"<= {config.max_method_failure_fraction}",
    })

    audit_ok = False
    if label_audit is not None:
        audit = label_audit.copy() if isinstance(label_audit, pd.DataFrame) else pd.read_csv(label_audit)
        if {"method", "all_trials_identical"}.issubset(audit.columns):
            row = audit.loc[audit["method"].astype(str).eq(config.candidate_method)]
            audit_ok = bool(len(row) and row["all_trials_identical"].astype(bool).all())
    checks.append({
        "name": "candidate_label_blindness_audit",
        "passed": audit_ok,
        "observed": audit_ok,
        "required": True,
    })
    checks.append({
        "name": "hyperparameters_frozen_before_external_benchmark",
        "passed": bool(frozen_hyperparameters),
        "observed": bool(frozen_hyperparameters),
        "required": True,
    })
    checks.append({
        "name": "canonical_source_lock_present",
        "passed": bool(canonical_source_lock),
        "observed": bool(canonical_source_lock),
        "required": True,
    })

    passed = all(bool(x["passed"]) for x in checks)
    return {
        "schema_version": 1,
        "passed": passed,
        "config": asdict(config),
        "checks": checks,
        "claim_guidance": (
            "Minimum provenance/evidence gate passed; statistical superiority claims still require "
            "pre-specified multi-dataset analyses and effect sizes."
            if passed
            else "Claim gate not passed: do not state that GSR-FS replaces, outperforms, or is state of the art."
        ),
    }


def write_publication_gate_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


@dataclass(frozen=True)
class SuperiorityGateConfig:
    """Conservative guardrail for strong outperformance/replacement claims.

    Passing the ordinary PublicationGate only means that enough auditable evidence
    exists to support a methodological paper.  This stricter gate is intentionally
    harder: it requires a significant multi-dataset omnibus result, the candidate
    to have the best mean rank, at least one multiplicity-corrected win against a
    canonical/author-code comparator, and no corrected significant loss.
    """
    metric: str = "balanced_accuracy"
    alpha: float = 0.05
    min_complete_datasets: int = 25
    min_significant_canonical_wins: int = 1
    candidate_method: str = "GSR-FS"
    require_best_mean_rank: bool = True
    require_significant_omnibus: bool = True


def evaluate_superiority_gate(results, *, config=None, publication_gate_report=None):
    """Evaluate whether evidence supports a strong superiority/replacement claim.

    This is an engineering guardrail, not a replacement for manuscript judgment.
    Positive effect direction means the candidate is better for the configured
    metric.  Only canonical/author-code comparators count toward the required wins.
    """
    from .stats import compare_methods

    config = config or SuperiorityGateConfig()
    df = results.copy() if isinstance(results, pd.DataFrame) else pd.read_csv(results)
    if config.metric not in df.columns:
        raise ValueError(f"Metric {config.metric!r} is not present in benchmark results.")

    res = compare_methods(
        df,
        config.metric,
        reference=config.candidate_method,
        higher_is_better=True,
        alpha=config.alpha,
    )
    checks = []

    evidence_ok = True
    if publication_gate_report is not None:
        if isinstance(publication_gate_report, (str, Path)):
            publication_gate_report = json.loads(Path(publication_gate_report).read_text(encoding="utf-8"))
        evidence_ok = bool(publication_gate_report.get("passed", False))
    checks.append({
        "name": "minimum_evidence_publication_gate",
        "passed": evidence_ok,
        "observed": evidence_ok,
        "required": True,
    })

    n_complete = int(res["n_complete_datasets"])
    checks.append({
        "name": "minimum_complete_datasets_for_superiority",
        "passed": n_complete >= config.min_complete_datasets,
        "observed": n_complete,
        "required": config.min_complete_datasets,
    })

    friedman_p = float(res["friedman_p"]) if np.isfinite(res["friedman_p"]) else np.nan
    omnibus_ok = bool(np.isfinite(friedman_p) and friedman_p < config.alpha)
    checks.append({
        "name": "significant_friedman_omnibus",
        "passed": (omnibus_ok if config.require_significant_omnibus else True),
        "observed": friedman_p,
        "required": f"p < {config.alpha}" if config.require_significant_omnibus else "optional",
    })

    ranks = res["mean_ranks"].copy()
    cand_row = ranks.loc[ranks["method"].astype(str).eq(config.candidate_method)]
    if cand_row.empty:
        raise ValueError(f"Candidate method {config.candidate_method!r} absent from complete comparison.")
    cand_rank = float(cand_row.iloc[0]["mean_rank"])
    best_rank = float(ranks["mean_rank"].min())
    best_ok = bool(np.isclose(cand_rank, best_rank) or cand_rank < best_rank)
    checks.append({
        "name": "candidate_best_mean_rank",
        "passed": (best_ok if config.require_best_mean_rank else True),
        "observed": cand_rank,
        "best_observed": best_rank,
        "required": "best mean rank" if config.require_best_mean_rank else "optional",
    })

    # Determine which competitors have externally sourced canonical/author code.
    source_col = None
    for c in ["implementation_source", "external_source", "source_kind"]:
        if c in df.columns:
            source_col = c
            break
    canonical = set()
    if source_col is not None:
        for method, sub in df.groupby("method"):
            vals = sub[source_col].dropna().astype(str).tolist()
            kinds = {_source_kind(v) if source_col != "source_kind" else str(v) for v in vals}
            if kinds & {"canonical", "author_code"}:
                canonical.add(str(method))

    post = res["posthoc"].copy()
    if not post.empty:
        post["canonical_competitor"] = post["competitor"].astype(str).isin(canonical)
        post["significant_win"] = (
            post["canonical_competitor"]
            & post["significant_holm"].astype(bool)
            & (pd.to_numeric(post["mean_oriented_difference"], errors="coerce") > 0)
        )
        post["significant_loss"] = (
            post["canonical_competitor"]
            & post["significant_holm"].astype(bool)
            & (pd.to_numeric(post["mean_oriented_difference"], errors="coerce") < 0)
        )
        n_wins = int(post["significant_win"].sum())
        n_losses = int(post["significant_loss"].sum())
    else:
        n_wins = n_losses = 0

    checks.append({
        "name": "holm_significant_canonical_wins",
        "passed": n_wins >= config.min_significant_canonical_wins,
        "observed": n_wins,
        "required": config.min_significant_canonical_wins,
    })
    checks.append({
        "name": "no_holm_significant_canonical_losses",
        "passed": n_losses == 0,
        "observed": n_losses,
        "required": 0,
    })

    passed = all(bool(c["passed"]) for c in checks)
    return {
        "schema_version": 1,
        "passed": passed,
        "config": asdict(config),
        "metric": config.metric,
        "checks": checks,
        "mean_ranks": ranks.to_dict(orient="records"),
        "posthoc": post.to_dict(orient="records") if not post.empty else [],
        "claim_guidance": (
            "Strong superiority guardrail passed; manuscript wording must still match effect sizes, scope, and limitations."
            if passed else
            "Strong superiority guardrail failed: do not claim that GSR-FS replaces, dominates, or is state of the art."
        ),
    }


def write_superiority_gate_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path
