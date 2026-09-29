from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def environment_manifest(extra=None):
    packages = {}
    for name in ["gsrfs", "numpy", "scipy", "scikit-learn", "pandas", "openml"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }
    if extra:
        out["configuration"] = extra
    return out


def stable_json_hash(obj):
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_manifest(path, extra=None):
    path = Path(path)
    payload = environment_manifest(extra=extra)
    payload["sha256"] = stable_json_hash(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
