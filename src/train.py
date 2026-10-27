"""
Train a language-driven grasp detector with one experiment.

Usage (the same command on the Mac, the PC, Colab and Kaggle; see terminal_commands.md):
    python src/train.py experiment=<EXP> [KEY=VALUE ...]
    python src/train.py experiment=<EXP> --cfg job                       # print the composed config, run nothing
    python src/train.py experiment=<EXP> +trainer.fast_dev_run=true      # one batch of train and val (smoke test)
    python src/train.py experiment=<EXP> ckpt_path=<RUN>/checkpoints/last.ckpt   # resume a stopped run

<EXP> is configs/experiment/<EXP>.yaml (only what differs from configs/train.yaml). The platform is detected
(src/utils/platform.py) and picks configs/paths/<platform>.yaml. A run writes into
<log_dir>/<EXP>/<YYYY-MM-DD>_<HH-MM-SS>/: .hydra/config.yaml (the full config used), metrics.csv, and checkpoints/
(last.ckpt with the full training state every epoch, and the best one by val/success).

Author: Duc Anh Hoang
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)  # src.* importable when run as a script
sys.dont_write_bytecode = True  # no __pycache__ in the code folder (on Colab it is read-only by rule)
os.environ.setdefault("PROJECT_ROOT", ROOT)  # read by configs/paths/*.yaml

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.utils.platform import add_paths_override  # noqa: E402


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    import lightning as L
    from hydra.core.hydra_config import HydraConfig
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    from src.data.grasp_anything import GraspAnythingData
    from src.models.grasp_module import GraspModule

    run_dir = HydraConfig.get().runtime.output_dir
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Batch":
        # A batch kernel has no terminal: the bar's lines flood the saved notebook and Kaggle's save step hangs.
        cfg.trainer.enable_progress_bar = False
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Run folder: {run_dir}", flush=True)
    L.seed_everything(cfg.seed, workers=True)
    data = GraspAnythingData(**cfg.data)
    model = GraspModule(**cfg.model, text_model=cfg.data.text_model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename="epoch{epoch:02d}-success{val/success:.4f}",
        auto_insert_metric_name=False,
        monitor="val/success",
        mode="max",
        save_last=True,
    )
    trainer = L.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        callbacks=[checkpoint],
        logger=CSVLogger(run_dir, name="", version=""),
        default_root_dir=run_dir,
    )
    trainer.fit(model, data, ckpt_path=cfg.ckpt_path)
    print(f"Best checkpoint: {checkpoint.best_model_path} (val/success {checkpoint.best_model_score})", flush=True)


if __name__ == "__main__":
    add_paths_override(sys.argv)
    main()
