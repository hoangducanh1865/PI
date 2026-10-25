"""
Every main config and every experiment composes and resolves on every platform, so a broken defaults list or
interpolation fails here and not at the start of a GPU session.

Author: Duc Anh Hoang
"""

import glob
import os

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "configs")
PLATFORMS = ("local", "colab", "kaggle")
EXPERIMENTS = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(CONFIG_DIR, "experiment", "*.yaml")))


def resolve(config_name: str, overrides: list[str]):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
    OmegaConf.resolve(cfg)
    return cfg


@pytest.fixture(autouse=True)
def project_root(monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", ROOT)


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_train_experiment(platform, experiment):
    cfg = resolve("train", [f"paths={platform}", f"experiment={experiment}"])
    assert cfg.name == experiment
    assert cfg.hydra.run.dir.startswith(f"{cfg.paths.log_dir}/{experiment}/")


@pytest.mark.parametrize("platform", PLATFORMS)
def test_eval(platform):
    cfg = resolve("eval", [f"paths={platform}", "ckpt_path=last.ckpt"])
    assert cfg.paths.root_dir == ROOT
