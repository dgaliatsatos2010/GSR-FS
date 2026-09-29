"""Static preflight for the networked v0.16 benchmark workflow."""
from __future__ import annotations
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github/workflows/cc18_v016.yml"

def main():
    checks = {}
    text = WF.read_text(encoding="utf-8")
    checks["workflow_exists"] = WF.exists()
    checks["canonical_commit_pinned"] = "48cffad4e88ff4b9d2f1c7baffb314d1b3303792" in text
    checks["finalizer_path_matches"] = "benchmarks/results/v16/cc18_${{ inputs.baseline_set }}" in text
    checks["artifact_path_v16"] = "path: benchmarks/results/v16/" in text
    checks["required_scripts_exist"] = all((ROOT / p).exists() for p in [
        "benchmarks/run_cc18.py",
        "benchmarks/finalize_cc18_v16.py",
        "benchmarks/run_fgmrw_cc18_v16.py",
        "CC18_TASKS.json",
        "CANONICAL_SOURCE_LOCK.json",
        "RECENT_AUTHOR_SOURCE_LOCK.json",
    ])
    lock = json.loads((ROOT / "RECENT_AUTHOR_SOURCE_LOCK.json").read_text(encoding="utf-8"))
    checks["fgmrw_not_vendored"] = lock.get("vendored_into_gsrfs") is False
    checks["fgmrw_label_blind"] = lock.get("strict_protocol", {}).get("labels_passed_to_selector") is False
    checks["overall_pass"] = all(checks.values())
    out = ROOT / "CI_PREFLIGHT_CURRENT_V016.json"
    out.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    raise SystemExit(0 if checks["overall_pass"] else 2)

if __name__ == "__main__":
    main()
