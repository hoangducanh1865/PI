"""
Generate the videos of the chosen Bridge test clips with one IRASim release model (IRASim-Frame-Ada, IRASim-Video-Ada,
LVDM or VDM), using the authors' own evaluation config with only the overrides below, and copy them out.

Usage (from video_eval/, after scripts/setup_irasim.sh and scripts/select_episodes.py):
    python scripts/run_irasim.py --irasim <IRASIM_DIR> --model <MODEL> --out <OUT_DIR> [--seed 3407] [--gpu 0]
        [--annotation annotation_<NAME> --name <MODEL>_<NAME>]   # perturbed actions, make_perturbed_annotations.py

Overrides of configs/evaluation/bridge/<MODEL>.yaml: annotation_name = annotation_subset (our clips; or --annotation),
global_seed (the initial noise; 3407 is the authors' value), anno (one result folder per model and seed),
local_val_batch_size (4, VDM keeps 1: one T4) and num_workers (2: 4 CPUs on Kaggle). Writes
<OUT_DIR>/<NAME>/seed<SEED>/<clip>.mp4 (<NAME> = --name, default <MODEL>) with <clip> = <episode>_<camera>_<start
frame>, 16 frames at 256 x 320, and the run log next to them.

Author: Duc Anh Hoang
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

from omegaconf import OmegaConf

MODELS = ("frame_ada", "video_ada", "lvdm", "vdm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--irasim", required=True, help="IRASim clone prepared by scripts/setup_irasim.sh")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--annotation", default="annotation_subset", help="annotation folder of the clips' actions")
    parser.add_argument("--name", help="output folder name (default: the model name)")
    args = parser.parse_args()

    config = OmegaConf.load(os.path.join(args.irasim, "configs", "evaluation", "bridge", f"{args.model}.yaml"))
    name = args.name or args.model
    anno = f"bridge_{name}_seed{args.seed}"
    overrides = {
        "annotation_name": args.annotation,
        "global_seed": args.seed,
        "anno": anno,
        "local_val_batch_size": 1 if args.model == "vdm" else 4,
        "num_workers": 2,
    }
    config = OmegaConf.merge(config, overrides)
    config_path = os.path.join(args.irasim, "configs", "ours", f"{anno}.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    OmegaConf.save(config, config_path)

    out = os.path.join(args.out, name, f"seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), WANDB_MODE="offline",
               PYTHONPATH=os.path.join(args.irasim, "pkgs"))  # fmt: skip
    command = ["torchrun", "--nproc_per_node", "1", "--master_port", str(29500 + args.gpu), "main.py",
               "--config", config_path]  # fmt: skip
    start = time.time()
    with open(os.path.join(out, "run.log"), "w") as log:
        code = subprocess.run(command, cwd=args.irasim, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    videos = glob.glob(os.path.join(args.irasim, "IRASim", "results", "*", "*", f"{anno}-debug", "checkpoints", "*",
                                    "test_sample_videos", "*.mp4"))  # fmt: skip
    for video in videos:
        shutil.copy(video, out)
    print(f"{name} seed {args.seed}: exit {code}, {len(videos)} videos in {time.time() - start:.0f} s -> {out}")
    sys.exit(code)


if __name__ == "__main__":
    main()
