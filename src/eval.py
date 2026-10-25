"""
Evaluate a checkpoint: a prediction succeeds if, for at least one ground-truth grasp, IoU > 25% and the angle differs
by less than 30 degrees; report the success rate on Seen and Unseen and their harmonic mean H.

Usage (the same command on the Mac, the PC, Colab and Kaggle; see terminal_commands.md):
    python src/eval.py ckpt_path=<CKPT> [KEY=VALUE ...]

The platform is detected (src/utils/platform.py) and picks configs/paths/<platform>.yaml.

Author: Duc Anh Hoang
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)  # src.* importable when run as a script
os.environ.setdefault("PROJECT_ROOT", ROOT)  # read by configs/paths/*.yaml

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.utils.platform import add_paths_override  # noqa: E402


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))  # the metrics come with the baseline


if __name__ == "__main__":
    add_paths_override(sys.argv)
    main()
