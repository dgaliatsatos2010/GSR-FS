from pathlib import Path
import numpy as np
import pandas as pd

from gsrfs import GSRSelector
from gsrfs.synthetic import null_independent, redundant_blocks, outlier_contamination
from gsrfs.metrics import group_coverage, relevant_precision, redundancy_rate

OUT = Path(__file__).resolve().parent / "results" / "v06"
OUT.mkdir(parents=True, exist_ok=True)


def fit_auto(X, seed, alpha=0.05):
    return GSRSelector(
        n_features="auto",
        n_permutations=100,
        alpha=alpha,
        max_pairs=5000,
        pair_sampler="balanced",
        robust=True,
        coverage_power=2.0,
        residual_power=0.5,
        allow_empty=True,
        support_engine="vectorized",
        residual_engine="auto",
        random_state=seed,
    ).fit(X)

rows=[]
for scenario, gen, count in [
    ("null_independent", null_independent, 50),
    ("redundant_blocks", redundant_blocks, 30),
    ("outlier_contamination", outlier_contamination, 15),
]:
    for seed in range(count):
        X, y, groups, name = gen(seed=seed)
        sel=fit_auto(X,seed)
        idx=sel.get_support(indices=True)
        rows.append({
            "scenario":name,"seed":seed,"n_selected":sel.n_features_selected_,
            "any_selected":int(sel.n_features_selected_>0),
            "selected":",".join(map(str,idx.tolist())),
            "auto_threshold":sel.auto_threshold_,
            "max_observed_effective_support":float(np.max(sel.effective_support_)),
            "group_coverage":np.nan if not groups else group_coverage(idx,groups),
            "relevant_precision":np.nan if not groups else relevant_precision(idx,groups),
            "redundancy_rate":np.nan if not groups else redundancy_rate(idx,groups),
            "pair_sampler":sel.pair_sampling_scheme_,
            "rng_policy":sel.rng_policy_,
        })

df=pd.DataFrame(rows)
df.to_csv(OUT/'auto_calibration_v06_raw.csv',index=False)
summary=df.groupby('scenario').agg(
    runs=('seed','count'),
    any_selection_rate=('any_selected','mean'),
    mean_n_selected=('n_selected','mean'),
    sd_n_selected=('n_selected','std'),
    mean_group_coverage=('group_coverage','mean'),
    mean_relevant_precision=('relevant_precision','mean'),
    mean_redundancy_rate=('redundancy_rate','mean'),
    mean_threshold=('auto_threshold','mean'),
)
summary.to_csv(OUT/'auto_calibration_v06_summary.csv')
print(summary.to_string())
