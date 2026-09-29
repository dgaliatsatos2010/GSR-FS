from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GateExtensionDataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    source: str
    source_file: str
    target_name: str
    panel_status: str = "gate_extension_v014"
    groups: np.ndarray | None = None
    notes: str = ""


def _pkg_root(package: str) -> Path:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise ImportError(f"Package {package!r} is not installed.")
    return Path(spec.origin).resolve().parent


def _factorize(s: pd.Series) -> np.ndarray:
    vals = pd.Series(s).reset_index(drop=True)
    codes, _ = pd.factorize(vals.astype(str), sort=True)
    if (codes < 0).any():
        raise ValueError("Target contains missing values after filtering.")
    return codes.astype(int)


def _require_class_support(name: str, y: np.ndarray, n_splits: int = 3):
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < n_splits:
        raise ValueError(f"{name}: smallest class has {counts.min()} observations (< {n_splits}).")


def load_gate_extension_panel_v014():
    """Seven frozen natural-label datasets added after the 23-dataset checkpoint.

    The panel is intentionally heterogeneous and uses no target-informed feature
    selection or preprocessing.  Direct target encodings/proxies are removed
    explicitly (e.g. birth weight when predicting the MASS `low` indicator).
    Grouped datasets retain subject/entity IDs only as CV groups, never as input
    features.
    """
    smroot = _pkg_root("statsmodels")
    p9root = _pkg_root("plotnine")
    plotlyroot = _pkg_root("plotly")
    out: list[GateExtensionDataset] = []

    # 1. UCI Automobile: insurance-risk symbol is the natural ordinal class.
    f = smroot / "gam/tests/results/autos.csv"
    df = pd.read_csv(f)
    df = df.loc[df["symbol"].notna()].reset_index(drop=True)
    y = _factorize(df["symbol"])
    X = df.drop(columns=["symbol"]).copy()
    _require_class_support("autos_symbol", y)
    out.append(GateExtensionDataset(
        "autos_symbol", X, y,
        source="UCI Automobile via statsmodels bundled GAM reference data",
        source_file=str(f), target_name="symbol",
        notes="Natural UCI insurance-risk rating. No target-derived columns retained.",
    ))

    # 2. MASS birthwt: low is defined by bwt < 2500g, so bwt is removed as a
    # direct deterministic target proxy; id is also removed.
    f = smroot / "genmod/tests/results/stata_lbw_glm.csv"
    df = pd.read_csv(f)
    df = df.loc[df["low"].notna()].reset_index(drop=True)
    y = _factorize(df["low"])
    X = df.drop(columns=[c for c in ["low", "bwt", "id"] if c in df.columns]).copy()
    _require_class_support("birthwt_low", y)
    out.append(GateExtensionDataset(
        "birthwt_low", X, y,
        source="MASS/Hosmer-Lemeshow birthwt via statsmodels bundled Stata reference data",
        source_file=str(f), target_name="low",
        notes="Direct proxy bwt removed because low is defined as birth weight <2.5kg.",
    ))

    # 3. Cattaneo maternal smoking. Use pretreatment/background covariates only;
    # all duplicate smoking encodings and birth-weight outcomes are removed.
    f = smroot / "treatment/tests/results/cataneo2.csv"
    df = pd.read_csv(f)
    # Uniform y-independent row cap for canonical dense spectral methods.
    if len(df) > 600:
        rng = np.random.default_rng(1414)
        keep = np.sort(rng.choice(len(df), size=600, replace=False))
        df = df.iloc[keep].reset_index(drop=True)
    y = _factorize(df["mbsmoke_"])
    drop = {
        "mbsmoke_", "mbsmoke", "msmoke",  # target encodings
        "bweight", "lbweight",              # post-exposure outcomes/direct consequence
        "fbaby_", "mmarried_", "prenatal1_", # duplicate encodings retained in original forms
    }
    X = df.drop(columns=[c for c in drop if c in df.columns]).copy()
    _require_class_support("cattaneo_smoking", y)
    out.append(GateExtensionDataset(
        "cattaneo_smoking", X, y,
        source="Stata cattaneo2 / Cattaneo (2010) via statsmodels bundled treatment reference data",
        source_file=str(f), target_name="mbsmoke_",
        notes="Uniform 600-row cap independent of y. Duplicate smoking encodings and birth-weight outcomes removed.",
    ))

    # 4. Epilepsy trial: treatment assignment is natural binary class. Repeated
    # measurements are grouped by subject and subject id is removed from X.
    f = smroot / "genmod/tests/results/epil.csv"
    df = pd.read_csv(f)
    y = _factorize(df["trt"])
    groups = df["subject"].to_numpy(copy=True)
    X = df.drop(columns=[c for c in ["trt", "subject", "Unnamed: 0"] if c in df.columns]).copy()
    _require_class_support("epil_treatment", y)
    out.append(GateExtensionDataset(
        "epil_treatment", X, y,
        source="MASS epil / Thall & Vail epilepsy trial via statsmodels bundled GEE reference data",
        source_file=str(f), target_name="trt", groups=groups,
        notes="Subject-grouped CV prevents repeated-measure leakage; subject id excluded from X.",
    ))

    # 5. Brader-Valentino-Suhay framing experiment: natural binary behavioral
    # outcome, using the same explanatory variables as the statsmodels example.
    f = smroot / "stats/tests/results/framing.csv"
    df = pd.read_csv(f)
    y = _factorize(df["cong_mesg"])
    cols = ["emo", "treat", "age", "educ", "gender", "income"]
    X = df[cols].copy()
    _require_class_support("framing_congress_message", y)
    out.append(GateExtensionDataset(
        "framing_congress_message", X, y,
        source="Brader, Valentino & Suhay (2008) framing experiment via statsmodels mediation reference data",
        source_file=str(f), target_name="cong_mesg",
        notes="Natural behavioral outcome; predictor set follows the documented statsmodels mediation example.",
    ))

    # 6. Gapminder: classify calendar year from development indicators while
    # generalizing to held-out countries. Uniformly select 60 countries independent
    # of target year, retain all years, and group CV by country.
    f = plotlyroot / "package_data/datasets/gapminder.csv.gz"
    df = pd.read_csv(f)
    countries = np.array(sorted(df["country"].dropna().unique().tolist()), dtype=object)
    rng = np.random.default_rng(1414)
    chosen = set(rng.choice(countries, size=min(60, len(countries)), replace=False).tolist())
    df = df.loc[df["country"].isin(chosen)].sort_values(["country", "year"]).reset_index(drop=True)
    y = _factorize(df["year"])
    groups = df["country"].astype(str).to_numpy(copy=True)
    X = df[["continent", "lifeExp", "pop", "gdpPercap"]].copy()
    _require_class_support("gapminder_year", y)
    out.append(GateExtensionDataset(
        "gapminder_year", X, y,
        source="Gapminder Five Year data bundled with Plotly",
        source_file=str(f), target_name="year", groups=groups,
        notes="60 countries selected uniformly independent of y; grouped CV holds out countries; identifiers excluded.",
    ))

    # 7. Midwest county demographics: state is a natural 5-class label. County
    # names/PID and derived categorical summaries are excluded; only demographic
    # measurements are retained.
    f = p9root / "data/midwest.csv"
    df = pd.read_csv(f)
    y = _factorize(df["state"])
    drop = {"state", "PID", "county", "category", "inmetro"}
    X = df.drop(columns=[c for c in drop if c in df.columns]).copy()
    _require_class_support("midwest_state", y)
    out.append(GateExtensionDataset(
        "midwest_state", X, y,
        source="Midwest county demographics bundled with plotnine",
        source_file=str(f), target_name="state",
        notes="Identifiers and derived categorical summaries excluded; demographic measurements retained.",
    ))

    return out
