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


def test_colab_wins_over_its_kaggle_folders(monkeypatch):
    import importlib.util
    import os

    monkeypatch.delenv("PLATFORM", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "google.colab" else None)
    monkeypatch.setattr(os.path, "isdir", lambda path: path == "/kaggle/input")  # Colab also has /kaggle/input
    assert detect_platform() == "colab"
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert detect_platform() == "kaggle"
