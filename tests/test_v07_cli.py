import json
from pathlib import Path

import numpy as np
import pandas as pd

from gsrfs.cli import main


def test_cli_csv_fixed_k(tmp_path, capsys):
    rng = np.random.default_rng(4)
    z = rng.normal(size=80)
    df = pd.DataFrame({
        "id": np.arange(80),
        "signal": z,
        "copy": z + rng.normal(scale=0.02, size=80),
        "noise": rng.normal(size=80),
    })
    inp = tmp_path / "x.csv"
    out = tmp_path / "selected.csv"
    rep = tmp_path / "report.json"
    df.to_csv(inp, index=False)

    rc = main([
        str(inp), "--exclude-columns", "id", "--n-features", "2",
        "--permutations", "10", "--max-pairs", "1000", "--seed", "3",
        "--output", str(out), "--report", str(rep),
    ])
    assert rc == 0
    report = json.loads(rep.read_text())
    assert report["selection_used_y"] is False
    assert report["n_features_selected"] == 2
    selected = pd.read_csv(out)
    assert selected.shape == (80, 2)
    assert set(selected.columns).issubset({"signal", "copy", "noise"})


def test_cli_rejects_nonnumeric_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3, 4], "label": ["x", "y", "x", "y"]})
    inp = tmp_path / "bad.csv"
    df.to_csv(inp, index=False)
    try:
        main([str(inp), "--n-features", "1"])
    except ValueError as exc:
        assert "nonnumeric" in str(exc).lower()
    else:
        raise AssertionError("Expected nonnumeric CSV to be rejected")


def test_cli_substitution_diagnostic(tmp_path, capsys):
    rng = np.random.default_rng(10)
    z = rng.normal(size=100)
    df = pd.DataFrame({
        "signal": z,
        "copy": z + rng.normal(scale=0.01, size=100),
        "noise": rng.normal(size=100),
    })
    inp = tmp_path / "x_groups.csv"
    rep = tmp_path / "groups.json"
    df.to_csv(inp, index=False)
    rc = main([
        str(inp), "--n-features", "1", "--permutations", "8",
        "--max-pairs", "1200", "--seed", "2",
        "--diagnose-substitutes", "--substitution-threshold", "0.85",
        "--report", str(rep),
    ])
    assert rc == 0
    report = json.loads(rep.read_text())
    assert report["version"] == "0.16.0"
    assert report["substitution_diagnostic"]["selection_changed_by_grouping"] is False
    groups = report["substitution_diagnostic"]["groups"]
    assert any(set(v) >= {"signal", "copy"} for v in groups.values())


def test_cli_structural_groups_output(tmp_path):
    rng = np.random.default_rng(21)
    z = rng.normal(size=110)
    df = pd.DataFrame({
        "signal": z,
        "copy": z + rng.normal(scale=0.01, size=110),
        "noise": rng.normal(size=110),
    })
    inp = tmp_path / "x_structural.csv"
    rep = tmp_path / "structural.json"
    df.to_csv(inp, index=False)
    rc = main([
        str(inp), "--n-features", "1", "--permutations", "8",
        "--max-pairs", "1200", "--seed", "4",
        "--structural-groups", "--report", str(rep),
    ])
    assert rc == 0
    report = json.loads(rep.read_text())
    out = report["structural_group_output"]
    assert out["creates_latent_features"] is False
    assert out["selection_changed_by_grouping"] is False
    assert out["summary"]["n_structural_groups"] == 1
    assert out["groups"][0]["output_feature"] in {"signal", "copy", "noise"}
