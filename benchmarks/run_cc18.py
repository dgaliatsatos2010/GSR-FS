"""Checkpointed OpenML-CC18 runner for GSR-FS v0.16 external benchmarking.

Strict rules
------------
* Every selector receives X only.
* Ground-truth y and class count are evaluation-only.
* The 72 CC18 task IDs are frozen in ``CC18_TASKS.json`` rather than selected
  opportunistically from whatever the server returns on a given day.
* ``--baseline-set canonical`` uses optional bridges to an independently
  installed scikit-feature package; third-party source is not vendored.
* GSR-FS hyperparameters remain frozen before external evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from gsrfs.realdata import RealDataset
from gsrfs.benchmark import run_real_benchmark_flexible
from gsrfs.repro import write_manifest
from gsrfs import (
    GSRSelector, MaxVarianceSelector, LaplacianScoreSelector, SPECSelector,
    MCFSSelector, ScikitFeatureLaplacianSelector, ScikitFeatureSPECSelector,
    ScikitFeatureMCFSSelector, FGMRWAuthorSelector, build_source_integrity_record,
    audit_source_integrity_manifest, write_source_integrity_audit,
)

ROOT = Path(__file__).resolve().parents[1]


def _encode_predictors(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    numeric = list(X.select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [c for c in X.columns if c not in numeric]

    for c in numeric:
        col = pd.to_numeric(X[c], errors="coerce")
        med = col.median()
        X[c] = col.fillna(0.0 if pd.isna(med) else med)

    if categorical:
        for c in categorical:
            X[c] = X[c].astype("string").fillna("__MISSING__")
        X = pd.get_dummies(X, columns=categorical, dummy_na=False, dtype=float)

    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median(axis=0, numeric_only=True)
    X = X.fillna(med).fillna(0.0)
    keep = X.nunique(dropna=False) > 1
    return X.loc[:, keep].astype(float)


def load_task(task_id, max_samples, max_features, seed):
    import openml

    task = openml.tasks.get_task(int(task_id), download_data=True, download_splits=False)
    ds = task.get_dataset()
    X, y = task.get_X_and_y(dataset_format="dataframe")
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    valid = y.notna().to_numpy()
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)

    # Strictly uniform subsampling: no stratification or label-informed sampling.
    row_sampled = False
    if len(X) > int(max_samples):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(X), size=int(max_samples), replace=False))
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
        row_sampled = True

    X = _encode_predictors(X)
    if X.shape[1] < 2:
        raise ValueError("fewer than two usable predictors after encoding")
    if X.shape[1] > int(max_features):
        raise ValueError(f"encoded feature count {X.shape[1]} exceeds cap {max_features}")

    y_codes = y.astype("category").cat.codes.to_numpy()
    meta = {
        "source_name": str(ds.name),
        "row_sampled": bool(row_sampled),
        "n_samples_encoded": int(X.shape[0]),
        "n_features_encoded": int(X.shape[1]),
        "n_classes_evaluation_only": int(len(np.unique(y_codes))),
    }
    return RealDataset(
        name=f"task{task_id}_{ds.name}",
        X=X.to_numpy(dtype=float),
        y=y_codes,
        feature_names=tuple(map(str, X.columns)),
    ), meta


def method_factories(args):
    base = {
        "GSR-FS": lambda k, seed: GSRSelector(
            n_features=k,
            n_permutations=args.permutations,
            max_pairs=args.max_pairs,
            pair_sampler="balanced",
            residual_power=0.5,
            random_state=seed,
            support_engine="vectorized",
            residual_engine="auto",
        ),
        "MaxVariance": lambda k, seed: MaxVarianceSelector(n_features=k),
    }
    if args.baseline_set == "development":
        base.update({
            "LaplacianScore-dev": lambda k, seed: LaplacianScoreSelector(n_features=k, n_neighbors=7),
            "SPEC-dev": lambda k, seed: SPECSelector(n_features=k, gamma=1.0, style=0, standardize=False),
            "MCFS-eigengap-dev": lambda k, seed: MCFSSelector(
                n_features=k, n_clusters="eigengap", n_neighbors=7,
                alpha=0.003, max_clusters=10, random_state=seed,
            ),
        })
    else:
        base.update({
            "LaplacianScore-scikit-feature": lambda k, seed: ScikitFeatureLaplacianSelector(n_features=k),
            "SPEC-scikit-feature": lambda k, seed: ScikitFeatureSPECSelector(n_features=k, style=0),
            # Fixed five is the external implementation's own default and is X/y independent.
            "MCFS-scikit-feature-k5": lambda k, seed: ScikitFeatureMCFSSelector(n_features=k, n_clusters=5),
        })
        if args.baseline_set == "canonical_recent":
            base["FGMRW-UFS-TKDE2026-author"] = lambda k, seed: FGMRWAuthorSelector(
                n_features=k, alpha=0.1, s=0.8
            )
    return base


def _load_frozen_tasks():
    obj = json.loads((ROOT / "CC18_TASKS.json").read_text(encoding="utf-8"))
    ids = [int(x) for x in obj["task_ids"]]
    if len(ids) != int(obj["n_tasks"]) or len(set(ids)) != len(ids):
        raise RuntimeError("Invalid frozen CC18 task manifest.")
    return obj, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument("--baseline-set", choices=["development", "canonical", "canonical_recent"], default="development")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        import openml  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install benchmark dependencies: pip install 'gsrfs[benchmark]'") from exc

    frozen, task_ids = _load_frozen_tasks()
    out = Path(__file__).resolve().parent / "results" / "v16" / f"cc18_{args.baseline_set}"
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "cc18_v15_raw.csv"
    manifest_path = out / "cc18_v15_dataset_manifest.csv"
    source_integrity_records = []

    config = {
        "suite": frozen["suite_name"],
        "suite_id": frozen["suite_id"],
        "frozen_task_count": len(task_ids),
        "limit": args.limit,
        "seeds": args.seeds,
        "gsr_permutations": args.permutations,
        "max_pairs": args.max_pairs,
        "pair_sampler": "balanced_round_robin",
        "max_samples_uniform": args.max_samples,
        "max_encoded_features": args.max_features,
        "gsr_residual_power_frozen": 0.5,
        "baseline_set": args.baseline_set,
        "selector_ground_truth_class_count_allowed": False,
        "labels_used_only_for_external_evaluation": True,
    }
    write_manifest(out / "environment_and_config.json", extra=config)
    (out / "task_ids_frozen.json").write_text(json.dumps(task_ids, indent=2), encoding="utf-8")

    completed = set()
    if args.resume and manifest_path.exists():
        old = pd.read_csv(manifest_path)
        completed = set(old.loc[old.status == "ok", "task_id"].astype(int).tolist())

    accepted = 0
    factories = method_factories(args)
    for task_id in task_ids:
        if accepted >= args.limit:
            break
        if int(task_id) in completed:
            continue
        t0 = time.perf_counter()
        record = {"task_id": int(task_id)}
        try:
            ds, meta = load_task(task_id, args.max_samples, args.max_features, seed=1729)
            result = run_real_benchmark_flexible(
                datasets=[ds], method_factories=factories, seeds=tuple(args.seeds)
            )
            # Treat comparator import/runtime failures as dataset-level failure in canonical mode.
            if result["error"].ne("").any():
                errors = result.loc[result.error.ne(""), ["method", "error"]].drop_duplicates()
                raise RuntimeError("benchmark method failure: " + errors.to_json(orient="records"))
            result.insert(0, "openml_task_id", int(task_id))
            if raw_path.exists() and args.resume:
                old_raw = pd.read_csv(raw_path)
                result = pd.concat([old_raw, result], ignore_index=True).drop_duplicates()
            result.to_csv(raw_path, index=False)
            accepted += 1
            record.update(meta)
            record.update({"status": "ok", "elapsed_seconds": time.perf_counter() - t0, "error": ""})
            source_integrity_records.append(build_source_integrity_record(
                dataset=ds.name,
                source_uri=f"https://www.openml.org/t/{int(task_id)}",
                source_kind="official_openml_task",
                raw_dataset_id=f"task:{int(task_id)}",
                target_name="evaluation_only",
                row_sampling_used_y=False,
                feature_filtering_used_y=False,
                preprocessing_used_y=False,
                imputation_used_y=False,
                scaling_used_y=False,
                encoding_used_y=False,
                dimension_reduction_used_y=False,
                provenance_complete=True,
                raw_feature_space_preserved=True,
                notes=(
                    "Task downloaded directly from OpenML. Optional row cap uses uniform X-index sampling "
                    "without y. Numeric median imputation and one-hot encoding are label-blind."
                ),
            ))
        except Exception as exc:
            record.update({
                "source_name": "", "row_sampled": np.nan,
                "n_samples_encoded": np.nan, "n_features_encoded": np.nan,
                "n_classes_evaluation_only": np.nan, "status": "skipped",
                "elapsed_seconds": time.perf_counter() - t0, "error": repr(exc),
            })

        new_man = pd.DataFrame([record])
        if manifest_path.exists() and args.resume:
            old_man = pd.read_csv(manifest_path)
            new_man = pd.concat([old_man, new_man], ignore_index=True).drop_duplicates(
                subset=["task_id"], keep="last"
            )
        new_man.to_csv(manifest_path, index=False)

    if source_integrity_records:
        audit = audit_source_integrity_manifest(source_integrity_records)
        write_source_integrity_audit(audit, out / "source_integrity_audit.json")

    print(f"Completed {accepted} new accepted datasets. Results: {raw_path}")


if __name__ == "__main__":
    main()
