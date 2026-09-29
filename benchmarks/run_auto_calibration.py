from pathlib import Path
import numpy as np
import pandas as pd

from gsrfs import GSRSelector
from gsrfs.synthetic import null_independent, redundant_blocks, outlier_contamination
from gsrfs.metrics import group_coverage, relevant_precision, redundancy_rate

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def fit_auto(X, seed, alpha=0.05):
    return GSRSelector(
        n_features="auto",
        n_permutations=100,
        alpha=alpha,
        max_pairs=5000,
        robust=True,
        coverage_power=2.0,
        residual_power=1.0,
        allow_empty=True,
        random_state=seed,
    ).fit(X)


rows = []
for seed in range(50):
    X, y, groups, name = null_independent(seed=seed)
    sel = fit_auto(X, seed)
    rows.append({
        "scenario": name,
        "seed": seed,
        "n_selected": sel.n_features_selected_,
        "any_selected": int(sel.n_features_selected_ > 0),
        "selected": ",".join(map(str, sel.get_support(indices=True).tolist())),
        "auto_threshold": sel.auto_threshold_,
        "max_observed_effective_support": float(np.max(sel.effective_support_)),
        "group_coverage": np.nan,
        "relevant_precision": np.nan,
        "redundancy_rate": np.nan,
    })

for seed in range(30):
    X, y, groups, name = redundant_blocks(seed=seed)
    sel = fit_auto(X, seed)
    idx = sel.get_support(indices=True)
    rows.append({
        "scenario": name,
        "seed": seed,
        "n_selected": sel.n_features_selected_,
        "any_selected": int(sel.n_features_selected_ > 0),
        "selected": ",".join(map(str, idx.tolist())),
        "auto_threshold": sel.auto_threshold_,
        "max_observed_effective_support": float(np.max(sel.effective_support_)),
        "group_coverage": group_coverage(idx, groups),
        "relevant_precision": relevant_precision(idx, groups),
        "redundancy_rate": redundancy_rate(idx, groups),
    })

# Deliberately difficult regime: coherent minority contamination. It is included
# as a diagnostic, not as a null, because an unlabeled method cannot in general
# distinguish a rare true subpopulation from a coherent outlier cluster.
for seed in range(15):
    X, y, groups, name = outlier_contamination(seed=seed)
    sel = fit_auto(X, seed)
    idx = sel.get_support(indices=True)
    rows.append({
        "scenario": name,
        "seed": seed,
        "n_selected": sel.n_features_selected_,
        "any_selected": int(sel.n_features_selected_ > 0),
        "selected": ",".join(map(str, idx.tolist())),
        "auto_threshold": sel.auto_threshold_,
        "max_observed_effective_support": float(np.max(sel.effective_support_)),
        "group_coverage": group_coverage(idx, groups),
        "relevant_precision": relevant_precision(idx, groups),
        "redundancy_rate": redundancy_rate(idx, groups),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "auto_calibration_raw.csv", index=False)
summary = df.groupby("scenario").agg(
    runs=("seed", "count"),
    any_selection_rate=("any_selected", "mean"),
    mean_n_selected=("n_selected", "mean"),
    sd_n_selected=("n_selected", "std"),
    mean_group_coverage=("group_coverage", "mean"),
    mean_relevant_precision=("relevant_precision", "mean"),
    mean_redundancy_rate=("redundancy_rate", "mean"),
    mean_threshold=("auto_threshold", "mean"),
)
summary.to_csv(OUT / "auto_calibration_summary.csv")
print(summary)
