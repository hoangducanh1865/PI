"""
Find the platform the code runs on (local, colab or kaggle), to pick configs/paths/<platform>.yaml.

"local" is the Mac or the PC. The PLATFORM environment variable overrides the detection. The checks look at the file
system and the installed packages, not at environment variables, so they also work in an SSH shell.

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
    if os.path.isdir("/kaggle/input"):
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
