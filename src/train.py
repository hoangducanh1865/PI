"""
Train a language-driven grasp detector with one experiment.

Usage (the same command on the Mac, the PC, Colab and Kaggle; see terminal_commands.md):
    python src/train.py experiment=<EXP> [KEY=VALUE ...]
    python src/train.py experiment=<EXP> --cfg job     # print the composed config, run nothing

<EXP> is configs/experiment/<EXP>.yaml (only what differs from configs/train.yaml). The platform is detected
(src/utils/platform.py) and picks configs/paths/<platform>.yaml. A run writes into
<log_dir>/<EXP>/<YYYY-MM-DD>_<HH-MM-SS>/ (.hydra/config.yaml = the full config used).

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


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))  # data, model and training loop come with the baseline


if __name__ == "__main__":
    add_paths_override(sys.argv)
    main()
