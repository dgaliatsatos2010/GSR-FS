import types
import numpy as np
import gsrfs.recent as recent
from gsrfs import FGMRWAuthorSelector


def test_git_blob_sha1_known_payload():
    assert recent.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_fgmrw_bridge_is_label_blind_and_complete(monkeypatch):
    seen = {}
    def fake_fgmrw(X, alpha, s):
        seen["X"] = np.array(X, copy=True)
        seen["alpha"] = alpha
        seen["s"] = s
        return np.arange(X.shape[1])[::-1]
    monkeypatch.setattr(recent, "_load_fgmrw", lambda cache_dir=None: types.SimpleNamespace(FGMRW_UFS=fake_fgmrw))
    X = np.array([[10., 1., 5.], [20., 2., 5.], [30., 4., 5.]])
    y1 = np.array([0, 0, 1])
    y2 = np.array([1, 0, 0])
    a = FGMRWAuthorSelector(n_features=1).fit(X, y1)
    b = FGMRWAuthorSelector(n_features=1).fit(X, y2)
    assert np.array_equal(a.get_support(indices=True), b.get_support(indices=True))
    # constant third column is excluded from author ranking and appended last
    assert a.ranking_indices_[-1] == 2
    assert seen["X"].min(axis=0).tolist() == [0.0, 0.0]
    assert seen["X"].max(axis=0).tolist() == [1.0, 1.0]
    assert seen["alpha"] == 0.1 and seen["s"] == 0.8
    assert a.selection_used_y_ is False
    assert a.selection_used_ground_truth_class_count_ is False


def test_recent_source_lock_is_pinned():
    assert recent.FGMRW_COMMIT == "325a904a28284a879c8a522ca7c4abc56449e9f5"
    assert recent.FGMRW_FILES["FGMRW-UFS-code.py"]["git_blob_sha1"] == "ff817880c7aa56366e957fe33e751b0db273443d"
    assert recent.FGMRW_FILES["GB.py"]["git_blob_sha1"] == "dd4814cdae5395ccaf35eb2c697e06efdbfb8bfb"
