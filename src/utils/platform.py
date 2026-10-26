"""
Find the platform the code runs on (local, colab or kaggle), to pick configs/paths/<platform>.yaml.

"local" is the Mac or the PC. The PLATFORM environment variable overrides the detection. Kaggle's image is built on
Colab's (it also has /content, google.colab and COLAB_* variables, and Colab has /kaggle/input), so the one reliable
sign of Kaggle is its own KAGGLE_KERNEL_RUN_TYPE variable (set in every Kaggle kernel and its subprocesses, never on
Colab); Colab is then recognised by its google.colab package, which also works in an SSH shell (seen on 2026-10-26).

Author: Duc Anh Hoang
"""

import importlib.util
import os

PLATFORMS = ("local", "colab", "kaggle")


def detect_platform() -> str:
    platform = os.environ.get("PLATFORM")
    if platform:
        if platform not in PLATFORMS:
            raise ValueError(f"PLATFORM={platform!r}, expected one of {PLATFORMS}")
        return platform
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    try:
        if importlib.util.find_spec("google.colab") is not None:
            return "colab"
    except ModuleNotFoundError:  # no "google" package at all
        pass
    return "local"


def add_paths_override(argv: list[str]) -> None:
    """Put paths=<platform> first in Hydra's command line, unless it already has a paths= override."""
    if not any(arg.startswith("paths=") for arg in argv[1:]):
        argv.insert(1, f"paths={detect_platform()}")
