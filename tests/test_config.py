from pathlib import Path

from rl2vla.utils.config import load_config


def test_config_inheritance(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("a:\n  b: 1\n  c: 2\nx: 3\n", encoding="utf-8")
    child.write_text("inherits: base.yaml\na:\n  b: 9\n", encoding="utf-8")
    config = load_config(child)
    assert config["a"]["b"] == 9
    assert config["a"]["c"] == 2
    assert config["x"] == 3
