import pandas as pd

from gsrfs import (
    build_source_integrity_record,
    audit_source_integrity_record,
    audit_source_integrity_manifest,
    evaluate_publication_gate,
    PublicationGateConfig,
)


def test_source_integrity_passes_raw_label_blind_source():
    rec = build_source_integrity_record(
        dataset="d1", source_uri="https://example.org/raw.csv",
        raw_dataset_id="1", target_name="class",
    )
    audit = audit_source_integrity_record(rec)
    assert audit["status"] == "PASS"
    assert audit["eligible"] is True


def test_source_integrity_rejects_supervised_feature_filtering():
    rec = build_source_integrity_record(
        dataset="d1", source_uri="https://example.org/reduced.pkl",
        feature_filtering_used_y=True,
        raw_feature_space_preserved=False,
        notes="Top features selected using RandomForestClassifier on y",
    )
    audit = audit_source_integrity_record(rec)
    assert audit["status"] == "FAIL"
    assert audit["eligible"] is False


def test_source_integrity_marks_unknown_provenance_review():
    rec = build_source_integrity_record(
        dataset="d1", source_uri="https://mirror.example/data.csv",
        provenance_complete=False,
    )
    audit = audit_source_integrity_record(rec)
    assert audit["status"] == "REVIEW"
    assert audit["eligible"] is False


def test_manifest_counts_only_pass_sources():
    manifest = [
        build_source_integrity_record(dataset="ok", source_uri="https://x/ok.csv"),
        build_source_integrity_record(dataset="bad", source_uri="https://x/bad.csv", row_sampling_used_y=True),
    ]
    audit = audit_source_integrity_manifest(manifest)
    assert audit["counts"] == {"PASS": 1, "FAIL": 1, "REVIEW": 0}
    assert audit["eligible_datasets"] == ["ok"]


def test_publication_gate_requires_integrity_audit():
    rows=[]
    for d in range(30):
        for method, kind in [("GSR-FS","native_or_local"),("C1","canonical"),("A2","author_code"),("C3","native_or_local")]:
            rows.append({"dataset": f"d{d}", "method":method, "nmi":0.5, "error":"", "source_kind":kind})
    df=pd.DataFrame(rows)
    label_audit=pd.DataFrame({"method":["GSR-FS"],"all_trials_identical":[True]})
    report=evaluate_publication_gate(df,label_audit=label_audit)
    assert report["passed"] is False
    failed={x["name"] for x in report["checks"] if not x["passed"]}
    assert "dataset_source_integrity_audit" in failed

    integrity=audit_source_integrity_manifest([
        build_source_integrity_record(dataset=f"d{i}", source_uri=f"https://official/{i}") for i in range(30)
    ])
    report2=evaluate_publication_gate(
        df,label_audit=label_audit,source_integrity=integrity,
        config=PublicationGateConfig(min_datasets=30,min_competitors=3,min_canonical_or_author_methods=2),
    )
    assert report2["passed"] is True


def test_failed_integrity_dataset_not_counted_toward_30():
    rows=[]
    for d in range(30):
        for method, kind in [("GSR-FS","native_or_local"),("C1","canonical"),("A2","author_code"),("C3","native_or_local")]:
            rows.append({"dataset": f"d{d}", "method":method, "nmi":0.5, "error":"", "source_kind":kind})
    integrity=[]
    for i in range(30):
        integrity.append(build_source_integrity_record(
            dataset=f"d{i}",source_uri=f"https://mirror/{i}",
            feature_filtering_used_y=(i==29),raw_feature_space_preserved=(i!=29),
        ))
    audit=audit_source_integrity_manifest(integrity)
    label_audit=pd.DataFrame({"method":["GSR-FS"],"all_trials_identical":[True]})
    report=evaluate_publication_gate(pd.DataFrame(rows),label_audit=label_audit,source_integrity=audit)
    check={x["name"]:x for x in report["checks"]}
    assert check["minimum_external_datasets"]["observed"] == 29
    assert report["passed"] is False
