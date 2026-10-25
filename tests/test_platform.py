"""
Platform detection: the PLATFORM override and the paths= override that src/train.py adds.

Author: Duc Anh Hoang
"""

import pytest

from src.utils.platform import add_paths_override, detect_platform


def test_platform_env_override(monkeypatch):
    monkeypatch.setenv("PLATFORM", "kaggle")
    assert detect_platform() == "kaggle"
    monkeypatch.setenv("PLATFORM", "aws")
    with pytest.raises(ValueError):
        detect_platform()


def test_paths_override_added_first(monkeypatch):
    monkeypatch.setenv("PLATFORM", "colab")
    argv = ["train.py", "experiment=baseline", "--cfg", "job"]
    add_paths_override(argv)
    assert argv == ["train.py", "paths=colab", "experiment=baseline", "--cfg", "job"]


def test_paths_override_kept(monkeypatch):
    monkeypatch.setenv("PLATFORM", "colab")
    argv = ["train.py", "paths=local"]
    add_paths_override(argv)
    assert argv == ["train.py", "paths=local"]
