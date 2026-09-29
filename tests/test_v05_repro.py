from gsrfs import environment_manifest, stable_json_hash


def test_stable_json_hash_is_order_invariant():
    assert stable_json_hash({"a": 1, "b": 2}) == stable_json_hash({"b": 2, "a": 1})


def test_environment_manifest_has_core_fields():
    m = environment_manifest({"seed": 1})
    assert "python" in m and "packages" in m
    assert m["configuration"]["seed"] == 1
