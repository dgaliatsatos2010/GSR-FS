from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/cc18_v016.yml"


def test_v16_workflow_is_pinned_and_paths_match_runner():
    text = WORKFLOW.read_text(encoding="utf-8")
    runner = (ROOT / "benchmarks/run_cc18.py").read_text(encoding="utf-8")
    assert "run_cc18.py" in text
    assert "finalize_cc18_v16.py" in text
    assert "run_fgmrw_cc18_v16.py" in text
    assert "48cffad4e88ff4b9d2f1c7baffb314d1b3303792" in text
    assert "benchmarks/results/v16/cc18_${{ inputs.baseline_set }}" in text
    assert '"v16" / f"cc18_{args.baseline_set}"' in runner
    assert "path: benchmarks/results/v16/" in text


def test_v16_workflow_uses_only_canonical_dropdown_and_pinned_recent_bridge():
    text = WORKFLOW.read_text(encoding="utf-8")
    # Recent comparator is executed in its own strict runner, not mixed into the
    # canonical dropdown.
    block = text.split("options:", 1)[1].split("jobs:", 1)[0]
    assert re.findall(r"^\s*-\s+(\S+)\s*$", block, flags=re.M) == ["canonical"]
    lock = json.loads((ROOT / "RECENT_AUTHOR_SOURCE_LOCK.json").read_text(encoding="utf-8"))
    assert lock["vendored_into_gsrfs"] is False
    assert lock["commit"] == "325a904a28284a879c8a522ca7c4abc56449e9f5"
    assert lock["strict_protocol"]["labels_passed_to_selector"] is False


def test_no_upstream_source_is_vendored_in_package_tree():
    bad = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("build/", "dist/", ".pytest_cache/")):
            continue
        low = rel.lower()
        if "/skfeature/" in f"/{low}" or low.endswith("fgmrw-ufs-code.py") or low.endswith("/gb.py"):
            bad.append(rel)
    assert bad == []


def test_v16_documentation_references_current_workflow():
    protocol = (ROOT / "STANDARDIZED_SUITE_PROTOCOL_V016.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert ".github/workflows/cc18_v016.yml" in protocol
    assert ".github/workflows/cc18_v015.yml" not in protocol
    assert ".github/workflows/cc18_v016.yml" in readme
    assert ".github/workflows/cc18_v015.yml" not in readme
