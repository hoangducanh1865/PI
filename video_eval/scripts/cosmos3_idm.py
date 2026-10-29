"""
Recover the robot actions of every clip with NVIDIA Cosmos3-Nano's inverse dynamics (no training of ours), then score
the clips in 3-D like scripts/idm.py.

Usage (from video_eval/; a vLLM-Omni server with nvidia/Cosmos3-Nano running, see scripts/setup_cosmos3.sh):
    python scripts/cosmos3_idm.py --gen <GEN_DIR> --models real still <MODEL_DIR> [...] [--url http://localhost:8000]

Requests follow the official vLLM-Omni recipe for Bridge: domain bridge_orig_lerobot, raw_action_dim 10, 640 x 480,
5 fps (Bridge's rate), 30 steps, guidance 1, flow shift 5, seed 0; our 16-frame clips give action_chunk_size 15 (it must
be num_frames or num_frames - 1). The answer is normalized and its 10 dimensions are not documented, so it is mapped to
our 7-D actions (IRASim's relative end-effector motion x 20 and gripper) by least squares fitted on real clips, with
2-fold cross-fitting: real clips are split by index parity, each half's map is fitted on the other half, and every
clip of every model uses the map of its half, so no clip is scored with a map fitted on its own real clip.
Writes <GEN_DIR>/<MODEL>/cosmos3_idm/<clip>.npz (raw answer) and <GEN_DIR>/metrics_idm_cosmos3.csv; the real
clips' scores are this IDM's own error.

Author: Duc Anh Hoang
"""

import argparse
import glob
import json
import os
import tempfile
import time

import imageio.v2 as imageio
import numpy as np
import requests
from idm import actions_from_states, trajectory_scores, write_rows
from PIL import Image

EXTRA = {"action_mode": "inverse_dynamics", "domain_name": "bridge_orig_lerobot", "raw_action_dim": 10,
         "action_chunk_size": 15}  # fmt: skip


def recover(url: str, path: str) -> np.ndarray:
    """Cosmos3-Nano's raw [15, 10] action answer for one 16-frame clip."""
    frames = [np.array(Image.fromarray(f).resize((640, 480), Image.BICUBIC)) for f in imageio.mimread(path)]
    with tempfile.NamedTemporaryFile(suffix=".mp4") as clip:
        imageio.mimwrite(clip.name, frames, fps=5)
        with open(clip.name, "rb") as f:
            form = {"model": "nvidia/Cosmos3-Nano", "prompt": "Recover the robot action trajectory from this clip.",
                    "size": "640x480", "num_frames": "16", "fps": "5", "num_inference_steps": "30",
                    "guidance_scale": "1.0", "flow_shift": "5.0", "seed": "0",
                    "extra_params": json.dumps(EXTRA)}  # fmt: skip
            job = requests.post(f"{url}/v1/videos", data=form, files={"input_reference": ("clip.mp4", f, "video/mp4")},
                                timeout=120).json()  # fmt: skip
    while True:
        status = requests.get(f"{url}/v1/videos/{job['id']}", timeout=30).json()
        if status["status"] == "completed":
            action = status["action"]
            return np.array(action["data"], dtype=np.float64).reshape(action["shape"])[-15:]
        if status["status"] in ("failed", "cancelled"):
            raise RuntimeError(f"{path}: {status}")
        time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="model folders; real is required (calibration)")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    raw = {}
    for name in args.models:
        folder = os.path.join(args.gen, name, "cosmos3_idm")
        os.makedirs(folder, exist_ok=True)
        for path in sorted(glob.glob(os.path.join(args.gen, name, "*.mp4"))):
            clip = os.path.basename(path)[:-4]
            cached = os.path.join(folder, f"{clip}.npz")
            if not os.path.exists(cached):
                np.savez(cached, raw=recover(args.url, path))
            raw[(name, clip)] = np.load(cached)["raw"]
        print(f"{name}: {sum(k[0] == name for k in raw)} clips recovered", flush=True)

    states = {c: np.load(os.path.join(args.gen, "states", f"{c}.npz")) for (n, c) in raw if n == "real"}
    clips = sorted(states)
    fold = {c: i % 2 for i, c in enumerate(clips)}
    maps = []
    for half in (0, 1):  # map applied to clips of fold `half`, fitted on the other fold
        train = [c for c in clips if fold[c] != half]
        x = np.concatenate([np.c_[raw[("real", c)], np.ones(15)] for c in train])
        y = np.concatenate([actions_from_states(states[c]["state"], states[c]["gripper"]) for c in train])
        maps.append(np.linalg.lstsq(x, y, rcond=None)[0])
    rows = []
    for (name, clip), answer in raw.items():
        if clip not in fold:
            continue
        pred = np.c_[answer, np.ones(15)] @ maps[fold[clip]]
        row, _ = trajectory_scores(pred, states[clip]["state"], states[clip]["gripper"])
        rows.append({"model": name, "clip": clip, **row})
    write_rows(os.path.join(args.gen, "metrics_idm_cosmos3.csv"), rows, args.models)


if __name__ == "__main__":
    main()
