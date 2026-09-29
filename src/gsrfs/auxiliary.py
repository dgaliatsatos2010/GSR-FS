from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class AuxiliaryDataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    source: str
    source_file: str
    target_name: str
    panel_status: str = "auxiliary_v014"
    groups: np.ndarray | None = None


def _pkg_root(package: str) -> Path:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise ImportError(f"Package {package!r} is not installed.")
    return Path(spec.origin).resolve().parent


def _factorize_target(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        vals = pd.Series(s).reset_index(drop=True)
        uniq = sorted(vals.dropna().unique().tolist())
        mapping = {v: i for i, v in enumerate(uniq)}
        return vals.map(mapping).to_numpy(dtype=int)
    codes, _ = pd.factorize(s.astype(str), sort=True)
    return codes.astype(int)


def _uniform_cap(df: pd.DataFrame, cap: int | None, seed: int = 1401) -> pd.DataFrame:
    if cap is None or len(df) <= int(cap):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(len(df), size=int(cap), replace=False))
    return df.iloc[idx].reset_index(drop=True)


def _finalize(name, df, target, *, source, source_file, drop=(), cap=None, derive=None):
    df = df.copy()
    if derive is None:
        mask = df[target].notna()
        df = df.loc[mask].reset_index(drop=True)
        y_series = df[target]
    else:
        base = derive["base"]
        mask = df[base].notna()
        df = df.loc[mask].reset_index(drop=True)
        if derive["op"] == "gt":
            y_series = (df[base].astype(float) > float(derive["value"])).astype(int)
        else:  # pragma: no cover
            raise ValueError("Unsupported target derivation")
    df = _uniform_cap(df, cap)
    if derive is None:
        y_series = df[target]
    else:
        base = derive["base"]
        y_series = (df[base].astype(float) > float(derive["value"])).astype(int)

    y = _factorize_target(pd.Series(y_series))
    drop_cols = set(drop)
    if derive is None:
        drop_cols.add(target)
    else:
        drop_cols.add(derive["base"])
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    if len(np.unique(y)) < 2:
        raise ValueError(f"{name}: target has fewer than 2 classes")
    return AuxiliaryDataset(
        name=name,
        X=X,
        y=y,
        source=source,
        source_file=str(source_file),
        target_name=target if derive is None else f"{derive['base']} > {derive['value']}",
    )


def load_auxiliary_natural_label_panel():
    """Frozen v0.14 auxiliary natural-label panel.

    All tasks use bundled public data.  Any row cap is uniform and independent of
    y.  Labels are returned for splitting/evaluation only and are never passed to
    feature selectors.
    """
    smroot = _pkg_root("statsmodels")
    p9root = _pkg_root("plotnine")
    grroot = _pkg_root("gradio")
    out = []

    f = smroot / "datasets/ccard/ccard.csv"
    out.append(_finalize("ccard_ownrent", pd.read_csv(f), "OWNRENT", source="statsmodels ccard", source_file=f))

    f = smroot / "datasets/heart/heart.csv"
    out.append(_finalize("heart_censor", pd.read_csv(f), "censors", source="statsmodels heart", source_file=f))

    f = smroot / "datasets/fair/fair.csv"
    out.append(_finalize(
        "fair_affair_any", pd.read_csv(f), "affair_any", source="statsmodels fair", source_file=f,
        cap=900, derive={"base": "affairs", "op": "gt", "value": 0},
    ))

    f = smroot / "duration/tests/results/bmt.csv"
    out.append(_finalize("bmt_status", pd.read_csv(f), "Status", source="statsmodels BMT", source_file=f))

    f = smroot / "miscmodels/tests/results/ologit_ucla.csv"
    out.append(_finalize("ologit_apply", pd.read_csv(f), "apply", source="statsmodels UCLA ordinal logit", source_file=f))

    f = smroot / "stats/tests/results/binary_constrict.csv"
    out.append(_finalize(
        "constrict_diagnosis", pd.read_csv(f), "diagnosis", source="statsmodels constrict", source_file=f,
        drop=("Case",),
    ))

    f = smroot / "discrete/tests/results/ships.csv"
    out.append(_finalize("ships_type", pd.read_csv(f), "ship", source="statsmodels ships", source_file=f))

    f = p9root / "data/mtcars.csv"
    out.append(_finalize(
        "mtcars_transmission", pd.read_csv(f), "am", source="plotnine mtcars", source_file=f,
        drop=("name",),
    ))

    f = p9root / "data/penguins.csv"
    out.append(_finalize("penguins_species", pd.read_csv(f), "species", source="plotnine penguins", source_file=f))

    f = p9root / "data/mpg.csv"
    out.append(_finalize("mpg_class", pd.read_csv(f), "class", source="plotnine mpg", source_file=f))

    f = p9root / "data/diamonds.csv"
    out.append(_finalize("diamonds_cut", pd.read_csv(f), "cut", source="plotnine diamonds", source_file=f, cap=900))

    f = p9root / "data/msleep.csv"
    out.append(_finalize(
        "msleep_vore", pd.read_csv(f), "vore", source="plotnine msleep", source_file=f,
        drop=("name",),
    ))

    f = grroot / "media_assets/data/titanic.csv"
    out.append(_finalize(
        "titanic_survived", pd.read_csv(f), "Survived", source="gradio bundled Titanic", source_file=f,
        drop=("PassengerId",),
    ))

    return out


def make_tabular_preprocessor(X: pd.DataFrame):
    """Create frozen label-blind mixed-type preprocessing for one training fold."""
    X = pd.DataFrame(X)
    num_cols = list(X.select_dtypes(include=[np.number, "bool"]).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median"))]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.02,
                    sparse_output=False,
                )),
            ]),
            cat_cols,
        ))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def encoded_k(p: int) -> int:
    p = int(p)
    if p <= 2:
        return p
    return min(p - 1, max(2, int(math.ceil(math.sqrt(p)))))
