from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from gsrfs.auxiliary import make_tabular_preprocessor, encoded_k
from gsrfs.downstream import canonical_offline_method_factories
from gsrfs.gate_extension import load_gate_extension_panel_v014

OUT = Path(__file__).resolve().parent / "results" / "v14" / "gate_extension"
OUT.mkdir(parents=True, exist_ok=True)


def _splits(ds):
    if ds.groups is None:
        sp = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        return list(sp.split(ds.X, ds.y)), "stratified"
    sp = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
    return list(sp.split(ds.X, ds.y, groups=ds.groups)), "stratified_group"


def run_one(ds):
    methods = canonical_offline_method_factories(gsr_permutations=12, max_pairs=8000)
    rows = []
    splits, split_kind = _splits(ds)
    for fold, (tr, te) in enumerate(splits):
        Xtr_raw = ds.X.iloc[tr].copy(); Xte_raw = ds.X.iloc[te].copy()
        ytr = ds.y[tr]; yte = ds.y[te]
        pre = make_tabular_preprocessor(Xtr_raw)
        Xtr = pre.fit_transform(Xtr_raw); Xte = pre.transform(Xte_raw)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr); Xte = scaler.transform(Xte)
        k = encoded_k(Xtr.shape[1])
        for method_name, factory in methods.items():
            selector = factory(k, 141400 + fold)
            t0 = time.perf_counter()
            try:
                selector.fit(Xtr)
                selected = np.asarray(selector.get_support(indices=True), dtype=int)
                fit_s = time.perf_counter() - t0
                if selected.size == 0:
                    raise RuntimeError("empty fixed-k subset")
                clf = LogisticRegression(max_iter=3000, class_weight="balanced")
                clf.fit(Xtr[:, selected], ytr)
                pred = clf.predict(Xte[:, selected])
                rows.append({
                    "dataset": ds.name, "panel_status": ds.panel_status, "source": ds.source,
                    "seed": 0, "fold": fold, "split_kind": split_kind,
                    "method": method_name,
                    "implementation_source": getattr(selector, "external_source_", "native_or_local"),
                    "n_samples": len(ds.y), "n_raw_features": ds.X.shape[1],
                    "n_encoded_features": Xtr.shape[1], "n_classes": len(np.unique(ds.y)),
                    "k": k, "selected": ",".join(map(str, selected.tolist())),
                    "balanced_accuracy": balanced_accuracy_score(yte, pred),
                    "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
                    "accuracy": accuracy_score(yte, pred),
                    "selector_fit_seconds": fit_s, "error": "",
                })
            except Exception as exc:
                rows.append({
                    "dataset": ds.name, "panel_status": ds.panel_status, "source": ds.source,
                    "seed": 0, "fold": fold, "split_kind": split_kind,
                    "method": method_name,
                    "implementation_source": ("scikit-feature:execution_failed" if "scikit-feature" in method_name else "native_or_local"),
                    "n_samples": len(ds.y), "n_raw_features": ds.X.shape[1],
                    "n_encoded_features": Xtr.shape[1], "n_classes": len(np.unique(ds.y)),
                    "k": k, "selected": "", "balanced_accuracy": np.nan, "macro_f1": np.nan,
                    "accuracy": np.nan, "selector_fit_seconds": time.perf_counter() - t0,
                    "error": repr(exc),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{ds.name}_raw.csv", index=False)
    print("\n", ds.name)
    print(df.groupby("method")[["balanced_accuracy", "macro_f1", "selector_fit_seconds"]].mean().sort_values("balanced_accuracy", ascending=False))
    return df


def main():
    dfs = [run_one(ds) for ds in load_gate_extension_panel_v014()]
    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(OUT / "gate_extension_canonical_raw.csv", index=False)


if __name__ == "__main__":
    main()
