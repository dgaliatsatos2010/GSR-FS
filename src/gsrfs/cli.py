from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .selector import GSRSelector
from .equivalence import GSRSubstitutionSelector
from .groups import GSRStructuralGroupSelector
from . import __version__


def _parse_n_features(value):
    if str(value).lower() == "auto":
        return "auto"
    try:
        k = int(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError("--n-features must be 'auto' or a positive integer") from exc
    if k < 1:
        raise argparse.ArgumentTypeError("--n-features must be 'auto' or a positive integer")
    return k


def _load_table(path, exclude_columns):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        exclude = [x for x in exclude_columns if x]
        missing = [x for x in exclude if x not in df.columns]
        if missing:
            raise ValueError(f"Excluded columns not found: {missing}")
        Xdf = df.drop(columns=exclude)
        nonnumeric = [c for c in Xdf.columns if not pd.api.types.is_numeric_dtype(Xdf[c])]
        if nonnumeric:
            raise ValueError(
                "GSR-FS CLI currently accepts numeric feature columns only. "
                f"Encode or exclude nonnumeric columns: {nonnumeric}"
            )
        return Xdf, df, "csv"
    if suffix == ".npy":
        X = np.load(path)
        if exclude_columns:
            raise ValueError("--exclude-columns is supported for CSV input only.")
        if np.asarray(X).ndim != 2:
            raise ValueError("NPY input must contain a 2D array.")
        return np.asarray(X, dtype=float), None, "npy"
    raise ValueError("Input must be .csv or .npy")


def build_parser():
    p = argparse.ArgumentParser(
        prog="gsrfs-select",
        description="Label-free GSR-FS feature selection for numeric CSV/NPY matrices.",
    )
    p.add_argument("input", help="Input .csv or .npy matrix")
    p.add_argument("--output", help="Selected matrix output (.csv or .npy)")
    p.add_argument("--report", help="JSON feature-selection report")
    p.add_argument("--exclude-columns", default="", help="Comma-separated CSV columns excluded before selection")
    p.add_argument("--n-features", type=_parse_n_features, default="auto")
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-pairs", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pair-sampler", choices=["balanced", "random"], default="balanced")
    p.add_argument("--diagnose-substitutes", action="store_true", help="Compute experimental v0.11 CGS substitution groups without changing selected features")
    p.add_argument("--substitution-threshold", type=float, default=0.85, help="CGS threshold used with substitution/group diagnostics")
    p.add_argument("--structural-groups", action="store_true", help="Return v0.12 structural-group output (representative + contextual substitutes) without changing selected features")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    excludes = [x.strip() for x in args.exclude_columns.split(",") if x.strip()]
    X, original_df, kind = _load_table(args.input, excludes)

    if args.structural_groups:
        selector_cls = GSRStructuralGroupSelector
        extra = {"substitution_threshold": args.substitution_threshold}
    elif args.diagnose_substitutes:
        selector_cls = GSRSubstitutionSelector
        extra = {"substitution_threshold": args.substitution_threshold}
    else:
        selector_cls = GSRSelector
        extra = {}
    selector = selector_cls(
        n_features=args.n_features,
        n_permutations=args.permutations,
        alpha=args.alpha,
        max_pairs=args.max_pairs,
        pair_sampler=args.pair_sampler,
        residual_power=0.5,
        random_state=args.seed,
        support_engine="vectorized",
        residual_engine="auto",
        **extra,
    )
    selector.fit(X)
    selected_idx = selector.get_support(indices=True)
    selected_names = selector.get_feature_names_out().tolist()

    report = {
        "method": "GSR-FS",
        "version": __version__,
        "input": str(Path(args.input).resolve()),
        "n_samples": int(np.asarray(X).shape[0]),
        "n_features_in": int(np.asarray(X).shape[1]),
        "n_features_selected": int(len(selected_idx)),
        "selected_indices": [int(x) for x in selected_idx],
        "selected_features": [str(x) for x in selected_names],
        "selection_used_y": False,
        "parameters": {
            "n_features": args.n_features,
            "n_permutations": args.permutations,
            "alpha": args.alpha,
            "max_pairs": args.max_pairs,
            "pair_sampler": args.pair_sampler,
            "residual_power": 0.5,
            "random_state": args.seed,
        },
    }

    if args.diagnose_substitutes or args.structural_groups:
        report["substitution_diagnostic"] = {
            "experimental": True,
            "threshold": float(args.substitution_threshold),
            "groups": selector.get_substitution_groups(names=True),
            "selection_changed_by_grouping": False,
        }

    if args.structural_groups:
        report["structural_group_output"] = {
            "experimental": True,
            "interpretive_only": True,
            "creates_latent_features": False,
            "selection_changed_by_grouping": False,
            "summary": selector.get_group_summary(),
            "groups": selector.get_structural_groups(),
        }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if kind == "csv":
            # Preserve selected feature names, but intentionally do not copy excluded
            # columns back into the transformed matrix: selection output is X_selected.
            transformed = pd.DataFrame(selector.transform(X), columns=selected_names)
            transformed.to_csv(out, index=False)
        else:
            np.save(out, selector.transform(X))

    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
