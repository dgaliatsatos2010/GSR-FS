from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


def aggregate_dataset_metric(df, metric, higher_is_better=True):
    """Aggregate repeated benchmark rows to one score per dataset and method.

    The function first averages over all repeated rows (e.g. seeds and fixed-k
    values) within each dataset/method. This prevents datasets with more repeated
    measurements from receiving extra weight in cross-dataset inference.
    """
    required = {"dataset", "method", metric}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = (
        df[["dataset", "method", metric]]
        .dropna(subset=[metric])
        .groupby(["dataset", "method"], as_index=False)[metric]
        .mean()
    )
    if out.empty:
        raise ValueError(f"No finite observations for metric {metric!r}.")
    out.attrs["higher_is_better"] = bool(higher_is_better)
    return out


def rank_methods_by_dataset(aggregated, metric, higher_is_better=True):
    """Return dataset-wise method ranks and their mean ranks.

    Rank 1 is always best. Ties receive average ranks.
    """
    pivot = aggregated.pivot(index="dataset", columns="method", values=metric)
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        raise ValueError("Need at least two complete datasets and two methods.")
    ranks = pivot.apply(
        lambda row: pd.Series(
            rankdata((-row if higher_is_better else row).to_numpy(), method="average"),
            index=row.index,
        ),
        axis=1,
    )
    ranks.index = pivot.index
    mean_ranks = ranks.mean(axis=0).sort_values()
    return pivot, ranks, mean_ranks


def _paired_rank_biserial(diff):
    """Paired rank-biserial correlation for nonzero paired differences."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff) & (diff != 0)]
    if diff.size == 0:
        return 0.0
    ranks = rankdata(np.abs(diff), method="average")
    pos = ranks[diff > 0].sum()
    neg = ranks[diff < 0].sum()
    denom = pos + neg
    return float((pos - neg) / denom) if denom else 0.0


def holm_adjust(p_values):
    """Holm step-down multiplicity adjustment."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for i, idx in enumerate(order):
        value = (m - i) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted


def compare_methods(
    df,
    metric,
    reference="GSR-FS",
    higher_is_better=True,
    alpha=0.05,
):
    """Multi-dataset nonparametric comparison with corrected post-hoc tests.

    Workflow
    --------
    1. Average repeated rows to one score per dataset/method.
    2. Keep only datasets complete across all methods.
    3. Compute dataset-wise ranks and a Friedman omnibus test.
    4. Compare ``reference`` with every competitor by paired Wilcoxon signed-rank
       tests and apply Holm correction.
    5. Report paired rank-biserial effect size, oriented so positive values mean
       the reference method is better.

    The function does not interpret statistical significance as practical
    superiority; raw paired differences and mean ranks are returned as well.
    """
    agg = aggregate_dataset_metric(df, metric, higher_is_better=higher_is_better)
    pivot, ranks, mean_ranks = rank_methods_by_dataset(
        agg, metric, higher_is_better=higher_is_better
    )
    methods = list(pivot.columns)
    if reference not in methods:
        raise ValueError(f"Reference method {reference!r} is not present.")
    if len(methods) < 3:
        friedman_stat, friedman_p = np.nan, np.nan
    else:
        samples = [pivot[m].to_numpy(dtype=float) for m in methods]
        friedman_stat, friedman_p = friedmanchisquare(*samples)
        friedman_stat, friedman_p = float(friedman_stat), float(friedman_p)

    rows = []
    raw_p = []
    ref = pivot[reference].to_numpy(dtype=float)
    for competitor in methods:
        if competitor == reference:
            continue
        other = pivot[competitor].to_numpy(dtype=float)
        oriented = (ref - other) if higher_is_better else (other - ref)
        if np.allclose(oriented, 0.0):
            stat, pval = 0.0, 1.0
        else:
            stat, pval = wilcoxon(oriented, zero_method="wilcox", alternative="two-sided")
            stat, pval = float(stat), float(pval)
        raw_p.append(pval)
        rows.append({
            "reference": reference,
            "competitor": competitor,
            "n_datasets": int(len(oriented)),
            "mean_reference": float(np.mean(ref)),
            "mean_competitor": float(np.mean(other)),
            "median_oriented_difference": float(np.median(oriented)),
            "mean_oriented_difference": float(np.mean(oriented)),
            "rank_biserial": _paired_rank_biserial(oriented),
            "wilcoxon_statistic": stat,
            "p_raw": pval,
        })

    adjusted = holm_adjust(raw_p) if raw_p else np.array([])
    for row, p_adj in zip(rows, adjusted):
        row["p_holm"] = float(p_adj)
        row["significant_holm"] = bool(p_adj < alpha)

    posthoc = pd.DataFrame(rows).sort_values(
        ["p_holm", "competitor"], na_position="last"
    ).reset_index(drop=True)
    mean_rank_df = mean_ranks.rename("mean_rank").reset_index().rename(columns={"index": "method"})

    return {
        "metric": metric,
        "higher_is_better": bool(higher_is_better),
        "n_complete_datasets": int(pivot.shape[0]),
        "methods": methods,
        "friedman_statistic": friedman_stat,
        "friedman_p": friedman_p,
        "mean_ranks": mean_rank_df,
        "posthoc": posthoc,
        "dataset_scores": pivot.reset_index(),
        "dataset_ranks": ranks.reset_index(),
    }
