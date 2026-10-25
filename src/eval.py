"""
Evaluate a checkpoint: a prediction succeeds if, for at least one ground-truth grasp, IoU > 25% and the angle differs
by less than 30 degrees; report the success rate on Seen and Unseen and their harmonic mean H.

Usage (the same command on the Mac, the PC, Colab and Kaggle; see terminal_commands.md):
    python src/eval.py ckpt_path=<CKPT> [KEY=VALUE ...]

The platform is detected (src/utils/platform.py) and picks configs/paths/<platform>.yaml. The text model is read from
the checkpoint. Writes into <log_dir>/eval/<YYYY-MM-DD>_<HH-MM-SS>/: results.json and, per split,
predictions_<split>.npz (keys, predicted grasps as x, y, w, h, theta with NaN when Q has no peak, success), the input
of the error analysis.

Author: Duc Anh Hoang
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)  # src.* importable when run as a script
sys.dont_write_bytecode = True  # no __pycache__ in the code folder (on Colab it is read-only by rule)
os.environ.setdefault("PROJECT_ROOT", ROOT)  # read by configs/paths/*.yaml

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.utils.platform import add_paths_override  # noqa: E402


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    import json

    import numpy as np
    import torch
    from hydra.core.hydra_config import HydraConfig

    from src.data.grasp_anything import GraspAnything
    from src.models.grasp_module import GraspModule, predict
    from src.utils.grasp import success

    run_dir = HydraConfig.get().runtime.output_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = GraspModule.load_from_checkpoint(cfg.ckpt_path, map_location=device)
    results = {"ckpt_path": cfg.ckpt_path}
    for split in ("test_seen", "test_unseen"):
        dataset = GraspAnything(cfg.data.dir, split, module.hparams.text_model)
        preds = predict(module.net, dataset, cfg.data.batch_size, cfg.data.num_workers)
        hits = np.array([success(p, dataset.grasps(i)) for i, p in enumerate(preds)])
        grasps = np.array([p if p is not None else np.full(5, np.nan) for p in preds], dtype=np.float32)
        np.savez(
            os.path.join(run_dir, f"predictions_{split}.npz"), keys=np.array(dataset.keys), grasps=grasps, success=hits
        )
        results[split] = float(hits.mean())
        print(f"{split}: success {results[split]:.4f} on {len(hits)} prompts", flush=True)
    seen, unseen = results["test_seen"], results["test_unseen"]
    results["H"] = 2 * seen * unseen / (seen + unseen) if seen + unseen else 0.0
    print(f"Seen {seen:.2f}  Unseen {unseen:.2f}  H {results['H']:.2f}", flush=True)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    add_paths_override(sys.argv)
    main()
