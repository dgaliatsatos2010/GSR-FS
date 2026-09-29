import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gsrfs import (
    array_fingerprint,
    build_external_ranking_record,
    load_external_ranking_record,
    selector_from_external_ranking,
    evaluate_publication_gate,
    PublicationGateConfig,
)


def test_external_ranking_roundtrip_and_fingerprint(tmp_path):
    X = np.arange(30, dtype=float).reshape(10, 3)
    fp = array_fingerprint(X, ["a", "b", "c"])
    rec = build_external_ranking_record(
        [2, 0, 1], method="AuthorMethod", dataset_id="toy",
        source="author/repo", feature_count=3, data_sha256=fp,
        selection_used_y=False, selection_used_ground_truth_class_count=False,
        source_commit="abc123",
    )
    path = tmp_path / "ranking.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    loaded = load_external_ranking_record(
        path, expected_p=3, expected_dataset_id="toy", expected_data_sha256=fp
    )
    assert loaded["ranking"] == [2, 0, 1]
    sel = selector_from_external_ranking(
        path, n_features=2, expected_p=3, expected_dataset_id="toy",
        expected_data_sha256=fp,
    ).fit(X)
    assert sel.get_support(indices=True).tolist() == [2, 0]
    assert sel.external_source_.startswith("author_code:")


def test_external_ranking_rejects_label_leakage(tmp_path):
    rec = build_external_ranking_record(
        [0, 1], method="Bad", dataset_id="toy", source="x",
        selection_used_y=True,
    )
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(ValueError, match="selection_used_y=false"):
        load_external_ranking_record(path, expected_p=2, require_label_blind=True)


def _mock_results(n_datasets=30, canonical_methods=2):
    rows = []
    methods = [
        ("GSR-FS", "native_or_local"),
        ("LocalControl", "native_or_local"),
        ("CanonicalA", "canonical" if canonical_methods >= 1 else "native_or_local"),
        ("AuthorB", "author_code" if canonical_methods >= 2 else "native_or_local"),
    ]
    for d in range(n_datasets):
        for method, kind in methods:
            rows.append({
                "dataset": f"d{d}", "method": method, "nmi": 0.5,
                "error": "", "source_kind": kind,
            })
    return pd.DataFrame(rows)


def test_publication_gate_fails_small_development_screen():
    audit = pd.DataFrame({"method": ["GSR-FS"], "all_trials_identical": [True]})
    report = evaluate_publication_gate(
        _mock_results(n_datasets=4, canonical_methods=0),
        label_audit=audit,
        frozen_hyperparameters=True,
        canonical_source_lock=True,
    )
    assert report["passed"] is False
    failed = {x["name"] for x in report["checks"] if not x["passed"]}
    assert "minimum_external_datasets" in failed
    assert "canonical_or_author_code_competitors" in failed


def test_publication_gate_can_pass_minimum_evidence_contract():
    audit = pd.DataFrame({"method": ["GSR-FS"], "all_trials_identical": [True]})
    from gsrfs import build_source_integrity_record, audit_source_integrity_manifest
    integrity = audit_source_integrity_manifest([
        build_source_integrity_record(dataset=f"d{i}", source_uri=f"https://official/{i}")
        for i in range(30)
    ])
    report = evaluate_publication_gate(
        _mock_results(n_datasets=30, canonical_methods=2),
        label_audit=audit,
        config=PublicationGateConfig(min_datasets=30, min_competitors=3, min_canonical_or_author_methods=2),
        frozen_hyperparameters=True,
        canonical_source_lock=True,
        source_integrity=integrity,
    )
    assert report["passed"] is True
