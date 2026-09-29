from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json

import pandas as pd


_TARGET_FLAGS = (
    "row_sampling_used_y",
    "feature_filtering_used_y",
    "preprocessing_used_y",
    "imputation_used_y",
    "scaling_used_y",
    "encoding_used_y",
    "dimension_reduction_used_y",
)


@dataclass(frozen=True)
class SourceIntegrityRecord:
    dataset: str
    source_uri: str
    source_kind: str = "official_or_raw"
    raw_dataset_id: str | None = None
    target_name: str | None = None
    source_sha256: str | None = None
    row_sampling_used_y: bool = False
    feature_filtering_used_y: bool = False
    preprocessing_used_y: bool = False
    imputation_used_y: bool = False
    scaling_used_y: bool = False
    encoding_used_y: bool = False
    dimension_reduction_used_y: bool = False
    provenance_complete: bool = True
    raw_feature_space_preserved: bool = True
    notes: str = ""


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_source_integrity_record(**kwargs):
    rec = SourceIntegrityRecord(**kwargs)
    return asdict(rec)


def _normalize_record(record):
    if isinstance(record, SourceIntegrityRecord):
        return asdict(record)
    if isinstance(record, dict):
        return dict(record)
    raise TypeError("record must be a dict or SourceIntegrityRecord")


def audit_source_integrity_record(record):
    """Audit whether a dataset source is eligible for label-blind UFS benchmarking.

    PASS means there is explicit provenance and no target-informed processing before
    the feature selector. FAIL means target information was used or the feature space
    was already supervised-filtered/reduced. REVIEW means provenance is insufficient.
    """
    r = _normalize_record(record)
    dataset = str(r.get("dataset", ""))
    reasons = []

    used_y = [flag for flag in _TARGET_FLAGS if bool(r.get(flag, False))]
    if used_y:
        reasons.append("target-informed pre-selector processing: " + ", ".join(used_y))

    if not bool(r.get("raw_feature_space_preserved", True)):
        reasons.append("raw feature space not preserved")

    provenance_complete = bool(r.get("provenance_complete", False))
    source_uri = str(r.get("source_uri", "")).strip()
    if not provenance_complete or not source_uri:
        status = "REVIEW"
        if reasons:
            status = "FAIL"
        elif not source_uri:
            reasons.append("missing source URI")
        if not provenance_complete:
            reasons.append("provenance incomplete")
    elif reasons:
        status = "FAIL"
    else:
        status = "PASS"

    return {
        "dataset": dataset,
        "status": status,
        "eligible": status == "PASS",
        "source_uri": source_uri,
        "source_kind": str(r.get("source_kind", "")),
        "raw_dataset_id": r.get("raw_dataset_id"),
        "target_name": r.get("target_name"),
        "source_sha256": r.get("source_sha256"),
        "reasons": reasons,
    }


def audit_source_integrity_manifest(manifest):
    if isinstance(manifest, (str, Path)):
        p = Path(manifest)
        if p.suffix.lower() == ".json":
            payload = json.loads(p.read_text(encoding="utf-8"))
            records = payload.get("records", payload) if isinstance(payload, dict) else payload
        else:
            records = pd.read_csv(p).to_dict(orient="records")
    elif isinstance(manifest, pd.DataFrame):
        records = manifest.to_dict(orient="records")
    elif isinstance(manifest, dict):
        records = manifest.get("records", [manifest])
    else:
        records = list(manifest)

    audits = [audit_source_integrity_record(r) for r in records]
    counts = {status: sum(a["status"] == status for a in audits) for status in ["PASS", "FAIL", "REVIEW"]}
    return {
        "schema_version": 1,
        "n_records": len(audits),
        "counts": counts,
        "eligible_datasets": sorted({a["dataset"] for a in audits if a["eligible"]}),
        "records": audits,
    }


def eligible_datasets_from_source_audit(audit):
    if isinstance(audit, (str, Path)):
        audit = json.loads(Path(audit).read_text(encoding="utf-8"))
    if "eligible_datasets" in audit:
        return set(map(str, audit["eligible_datasets"]))
    return {str(r["dataset"]) for r in audit.get("records", []) if bool(r.get("eligible", False))}


def write_source_integrity_audit(audit, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return path
