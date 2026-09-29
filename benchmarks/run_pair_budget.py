"""Pair-budget convergence study for the frozen v0.5 GSR-FS objective.

v0.6 separates pair-sampling and permutation RNG streams, so differences across
max_pairs values isolate pair-geometry approximation rather than changing the
permutation null at the same time.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gsrfs import GSRSelector
from gsrfs.metrics import jaccard
from gsrfs.realdata import load_builtin_benchmarks


def run(budgets=(500, 1000, 2500, 5000, 10000, 20000, 50000), permutations=20, seed=0):
    rows = []
    datasets = load_builtin_benchmarks(include_digits=False)
    for ds in datasets:
        X = ds.X
        p = X.shape[1]
        k = min(max(2, int(round(0.25 * p))), min(10, p))
        ref_budget = int(max(budgets))
        common = dict(
            n_features=k,
            n_permutations=int(permutations),
            random_state=int(seed),
            support_engine="vectorized",
            residual_engine="auto",
        )
        t0 = time.perf_counter()
        ref = GSRSelector(max_pairs=ref_budget, **common).fit(X)
        ref_time = time.perf_counter() - t0
        ref_sel = ref.get_support(indices=True)
        ref_rank = ref.ranking_.copy()

        for budget in budgets:
            if int(budget) == ref_budget:
                sel = ref
                elapsed = ref_time
            else:
                t0 = time.perf_counter()
                sel = GSRSelector(max_pairs=int(budget), **common).fit(X)
                elapsed = time.perf_counter() - t0
            chosen = sel.get_support(indices=True)
            rank = sel.ranking_.copy()
            rho = float(spearmanr(ref_rank, rank).statistic)
            rows.append({
                "dataset": ds.name,
                "n_samples": X.shape[0],
                "n_features": p,
                "k": k,
                "max_pairs": int(budget),
                "reference_pairs": ref_budget,
                "jaccard_selected": float(jaccard(ref_sel, chosen)),
                "rank_spearman": rho,
                "fit_seconds": elapsed,
                "reference_seconds": ref_time,
                "runtime_ratio_vs_reference": elapsed / max(ref_time, 1e-12),
                "same_permutation_bank": bool(sel.permutation_bank_sha256_ == ref.permutation_bank_sha256_),
                "selected": ",".join(map(str, chosen.tolist())),
                "reference_selected": ",".join(map(str, ref_sel.tolist())),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", nargs="+", type=int, default=[500,1000,2500,5000,10000,20000,50000])
    args = ap.parse_args()
    out = Path(__file__).resolve().parent / "results" / "v06"
    out.mkdir(parents=True, exist_ok=True)
    df = run(tuple(args.budgets), args.permutations, args.seed)
    df.to_csv(out / "pair_budget_v06_raw.csv", index=False)
    summary = df.groupby("max_pairs")[["jaccard_selected","rank_spearman","runtime_ratio_vs_reference"]].mean().reset_index()
    summary.to_csv(out / "pair_budget_v06_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
