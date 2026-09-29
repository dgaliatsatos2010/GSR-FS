"""Optional OpenML benchmark runner for broad external validation.

Example
-------
python benchmarks/run_openml.py --dataset-ids 61 187 1510

This script is intentionally not part of the core dependency set. Install the
benchmark extra first: ``pip install gsrfs[benchmark]``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from gsrfs.realdata import RealDataset
from gsrfs.benchmark import run_real_benchmark, summarize_real


def load_openml_numeric(dataset_id):
    try:
        import openml
    except ImportError as exc:
        raise SystemExit("Install the benchmark extra: pip install gsrfs[benchmark]") from exc

    ds = openml.datasets.get_dataset(int(dataset_id), download_data=True)
    target = ds.default_target_attribute
    if target is None:
        raise ValueError(f"OpenML dataset {dataset_id} has no default target attribute.")
    X, y, _, _ = ds.get_data(target=target, dataset_format="dataframe")
    X = X.select_dtypes(include=[np.number]).copy()
    if X.shape[1] < 2:
        raise ValueError(f"OpenML dataset {dataset_id} has fewer than two numeric predictors.")
    valid = X.notna().all(axis=1) & pd.Series(y).notna().to_numpy()
    X = X.loc[valid]
    y = pd.Series(y).loc[valid].astype("category").cat.codes.to_numpy()
    return RealDataset(
        name=f"openml_{dataset_id}_{ds.name}",
        X=X.to_numpy(dtype=float),
        y=y,
        feature_names=tuple(map(str, X.columns)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-ids", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--max-pairs", type=int, default=12000)
    args = parser.parse_args()

    datasets = [load_openml_numeric(i) for i in args.dataset_ids]
    df = run_real_benchmark(
        datasets=datasets,
        seeds=tuple(args.seeds),
        gsr_permutations=args.permutations,
        max_pairs=args.max_pairs,
    )
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "openml_raw.csv", index=False)
    summarize_real(df).to_csv(out / "openml_summary.csv")
    print(summarize_real(df))


if __name__ == "__main__":
    main()
