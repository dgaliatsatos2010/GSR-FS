from pathlib import Path
import tomllib

import gsrfs


def test_version_coherence():
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    assert data["project"]["version"] == "0.16.0"
    assert gsrfs.__version__ == "0.16.0"
    assert data["project"]["scripts"]["gsrfs-select"] == "gsrfs.cli:main"
