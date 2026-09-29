from __future__ import annotations

from pathlib import Path
import pandas as pd

from gsrfs.stats import compare_methods


def save_result(res, out, prefix):
    pd.DataFrame([{
        "metric": res["metric"],
        "higher_is_better": res["higher_is_better"],
        "n_complete_datasets": res["n_complete_datasets"],
        "friedman_statistic": res["friedman_statistic"],
        "friedman_p": res["friedman_p"],
    }]).to_csv(out / f"{prefix}_omnibus.csv", index=False)
    res["mean_ranks"].to_csv(out / f"{prefix}_mean_ranks.csv", index=False)
    res["posthoc"].to_csv(out / f"{prefix}_posthoc.csv", index=False)
    res["dataset_scores"].to_csv(out / f"{prefix}_dataset_scores.csv", index=False)
    res["dataset_ranks"].to_csv(out / f"{prefix}_dataset_ranks.csv", index=False)


def main():
    base = Path(__file__).resolve().parent
    results = base / "results"
    results.mkdir(exist_ok=True)
    path = results / "real_tabular_raw.csv"
    if not path.exists():
        path = results / "real_builtin_raw.csv"
    df = pd.read_csv(path)
    for metric, hib in [
        ("nmi", True),
        ("ari", True),
        ("knn_preservation", True),
        ("geometry_redundancy", False),
        ("normalized_effective_rank", True),
    ]:
        if metric not in df.columns:
            continue
        res = compare_methods(df, metric, reference="GSR-FS", higher_is_better=hib)
        save_result(res, results, f"stats_{metric}")
        print("\n", metric)
        print(res["mean_ranks"].to_string(index=False))
        print(res["posthoc"].to_string(index=False))


if __name__ == "__main__":
    main()
