from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .baselines import ExternalRankingSelector


SCHEMA_VERSION = 1


def array_fingerprint(X, feature_names=None):
    """Return a deterministic SHA-256 fingerprint for benchmark provenance.

    The fingerprint includes array shape, float64 C-order bytes, and optional
    feature names. It is intended to catch accidental dataset/column mismatches;
    it is not a privacy-preserving hash for sensitive data.
    """
    arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    h = hashlib.sha256()
    h.update(str(tuple(arr.shape)).encode("utf-8"))
    h.update(arr.tobytes(order="C"))
    if feature_names is not None:
        names = [str(x) for x in feature_names]
        h.update(json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def build_external_ranking_record(
    ranking,
    *,
    method,
    dataset_id,
    source,
    feature_count=None,
    data_sha256=None,
    selection_used_y=False,
    selection_used_ground_truth_class_count=False,
    source_version=None,
    source_commit=None,
    notes=None,
):
    ranking = np.asarray(ranking, dtype=int).ravel()
    if feature_count is None:
        feature_count = int(ranking.size)
    record = {
        "schema_version": SCHEMA_VERSION,
        "method": str(method),
        "dataset_id": str(dataset_id),
        "feature_count": int(feature_count),
        "indexing": "zero_based",
        "ranking": ranking.tolist(),
        "selection_used_y": bool(selection_used_y),
        "selection_used_ground_truth_class_count": bool(selection_used_ground_truth_class_count),
        "source": {
            "name": str(source),
            "version": None if source_version is None else str(source_version),
            "commit": None if source_commit is None else str(source_commit),
        },
        "data_sha256": None if data_sha256 is None else str(data_sha256),
        "notes": None if notes is None else str(notes),
    }
    validate_external_ranking_record(record, expected_p=feature_count, require_label_blind=False)
    return record


def validate_external_ranking_record(
    record,
    *,
    expected_p=None,
    expected_dataset_id=None,
    expected_data_sha256=None,
    require_label_blind=True,
):
    if not isinstance(record, dict):
        raise TypeError("External ranking record must be a JSON object/dict.")
    if int(record.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported external-ranking schema_version; expected {SCHEMA_VERSION}.")
    if record.get("indexing") != "zero_based":
        raise ValueError("External rankings must use explicit zero_based indexing.")

    p = int(record.get("feature_count", -1))
    if p < 1:
        raise ValueError("feature_count must be a positive integer.")
    ranking = np.asarray(record.get("ranking", []), dtype=int).ravel()
    if ranking.size != p or set(ranking.tolist()) != set(range(p)):
        raise ValueError("ranking must be a complete permutation of 0..feature_count-1.")
    if expected_p is not None and p != int(expected_p):
        raise ValueError(f"feature_count mismatch: record={p}, expected={int(expected_p)}.")
    if expected_dataset_id is not None and str(record.get("dataset_id")) != str(expected_dataset_id):
        raise ValueError("dataset_id does not match the benchmark dataset.")
    if expected_data_sha256 is not None and str(record.get("data_sha256")) != str(expected_data_sha256):
        raise ValueError("data_sha256 does not match the benchmark matrix.")

    source = record.get("source")
    if not isinstance(source, dict) or not str(source.get("name", "")).strip():
        raise ValueError("source.name is required for provenance.")

    if require_label_blind:
        if record.get("selection_used_y") is not False:
            raise ValueError("Publication protocol requires selection_used_y=false.")
        if record.get("selection_used_ground_truth_class_count") is not False:
            raise ValueError(
                "Publication protocol requires selection_used_ground_truth_class_count=false."
            )
    return True


def write_external_ranking_record(record, path):
    validate_external_ranking_record(record, require_label_blind=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_external_ranking_record(
    path,
    *,
    expected_p=None,
    expected_dataset_id=None,
    expected_data_sha256=None,
    require_label_blind=True,
):
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    validate_external_ranking_record(
        record,
        expected_p=expected_p,
        expected_dataset_id=expected_dataset_id,
        expected_data_sha256=expected_data_sha256,
        require_label_blind=require_label_blind,
    )
    return record


def selector_from_external_ranking(
    path,
    *,
    n_features,
    expected_p=None,
    expected_dataset_id=None,
    expected_data_sha256=None,
    require_label_blind=True,
):
    record = load_external_ranking_record(
        path,
        expected_p=expected_p,
        expected_dataset_id=expected_dataset_id,
        expected_data_sha256=expected_data_sha256,
        require_label_blind=require_label_blind,
    )
    src = record["source"]["name"]
    if record["source"].get("commit"):
        src += "@" + str(record["source"]["commit"])
    return ExternalRankingSelector(
        ranking=record["ranking"],
        n_features=n_features,
        source=f"author_code:{src}",
    )
