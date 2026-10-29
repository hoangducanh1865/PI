"""
Recover the robot actions of every clip with NVIDIA Cosmos3-Nano's inverse dynamics (no training of ours), then score
the clips in 3-D like scripts/idm.py.

Usage (from video_eval/; a vLLM-Omni server with nvidia/Cosmos3-Nano running, see scripts/setup_cosmos3.sh):
    python scripts/cosmos3_idm.py --gen <GEN_DIR> --models real still <MODEL_DIR> [...] [--url http://localhost:8000]

Requests follow the official vLLM-Omni recipe for Bridge (recipes/cosmos3/Cosmos3-Nano.md): domain bridge_orig_lerobot,
raw_action_dim 10, 640 x 480, 5 fps (Bridge's rate), 30 steps, guidance 1, flow shift 5, seed 0; our 16-frame clips
give action_chunk_size 15 (it must be num_frames or num_frames - 1).
The answer is decoded as NVIDIA's cosmos-framework encodes Bridge actions (github.com/NVIDIA/cosmos-framework, commit
1a2f03a: data/generator/action/datasets/bridge_orig_lerobot_dataset.py, action_normalization.py, utils/pose_utils.py,
normalizer_stats/bridge_orig_lerobot_stats.json), with nothing fitted:
- 10 numbers per step = [translation (3), rot6d (6), gripper (1)], quantile-normalized:
  a = (x + 1) / 2 (q99 - q01) + q01;
- translation and rotation are the step's relative pose T_i^-1 T_(i+1) ("backward_framewise"); rot6d = the first two
  columns of the rotation matrix (the third is their cross product, then projected onto a rotation by SVD);
- poses are taken in NVIDIA's frame: the Bridge state pose times fixed rotations (_DEFAULT_ROTATION, then
  _BRIDGE_TO_OPENCV) and moved 9.36 cm back along the gripper (_TCP_TO_FLANGE);
- IRASim's Bridge states store the roll of LeRobot's bridge_orig_lerobot (the data NVIDIA trained on) plus pi
  (checked 2026-10-29: roll median 2.47 rad in our 105 clips, 0.00 in 7 LeRobot episodes; pitch, yaw, xyz agree), so
  R_LeRobot = R_IRASim Rx(pi).
The gripper number is the commanded gripper NVIDIA trains on (LeRobot action[6]: binary, 1 = open; IRASim stores the
same command as action[:, 6]). It is not comparable with IRASim's continuous_gripper_state, which measures closing
(0 = open; r = -0.79 with the binary state over the 69 test episodes), and it is not scored here.
From the real pose at frame 0, the decoded steps give 16 poses, which are mapped back to IRASim's state convention and
scored by idm.trajectory_scores (via idm.actions_from_states), like our own IDMs.
Writes <GEN_DIR>/<MODEL>/cosmos3_idm/<clip>.npz (raw answer) and <GEN_DIR>/metrics_idm_cosmos3.csv; the real clips'
scores are this IDM's own error.

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
from scipy.spatial.transform import Rotation

EXTRA = {"action_mode": "inverse_dynamics", "domain_name": "bridge_orig_lerobot", "raw_action_dim": 10,
         "action_chunk_size": 15}  # fmt: skip
Q01 = np.array([-0.038884, -0.028667, -0.03784, 0.976292, -0.163098, -0.081545, -0.160193, 0.976322, -0.078872, 0.0])
Q99 = np.array([0.039722, 0.029068, 0.026702, 1.0, 0.160195, 0.081655, 0.163227, 1.0, 0.095189, 1.0])
# IRASim state rotation -> NVIDIA's kinematics frame (roll + pi, then _DEFAULT_ROTATION), and that frame -> OpenCV
TO_KINEMATICS = np.diag([1.0, -1.0, -1.0]) @ np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
TO_OPENCV = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
FLANGE = np.array([-0.093575, 0.0, 0.0])  # gripper_link in the kinematics frame of ee_gripper_link


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


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    m = np.stack([r6[:3], r6[3:], np.cross(r6[:3], r6[3:])], axis=1)
    u, _, vt = np.linalg.svd(m)
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


def decode(raw: np.ndarray, first: np.ndarray) -> np.ndarray:
    """[16, 6] poses (xyz, rpy) in IRASim's convention from the raw [15, 10] answer and the real first pose."""
    steps = (raw + 1) / 2 * (Q99 - Q01) + Q01
    rot = Rotation.from_euler("xyz", first[3:6]).as_matrix() @ TO_KINEMATICS
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rot @ TO_OPENCV, first[:3] + rot @ FLANGE
    states = [first]
    for step in steps:
        delta = np.eye(4)
        delta[:3, :3], delta[:3, 3] = rot6d_to_matrix(step[3:9]), step[:3]
        pose = pose @ delta
        rot = pose[:3, :3] @ TO_OPENCV.T
        state_rot = rot @ TO_KINEMATICS.T
        states.append(np.r_[pose[:3, 3] - rot @ FLANGE, Rotation.from_matrix(state_rot).as_euler("xyz")])
    return np.array(states)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    rows = []
    for name in args.models:
        folder = os.path.join(args.gen, name, "cosmos3_idm")
        os.makedirs(folder, exist_ok=True)
        for path in sorted(glob.glob(os.path.join(args.gen, name, "*.mp4"))):
            clip = os.path.basename(path)[:-4]
            cached = os.path.join(folder, f"{clip}.npz")
            if not os.path.exists(cached):
                np.savez(cached, raw=recover(args.url, path))
            raw = np.load(cached)["raw"]
            states = np.load(os.path.join(args.gen, "states", f"{clip}.npz"))
            state, gripper = states["state"], states["gripper"]
            pred = actions_from_states(decode(raw, state[0]), gripper)
            pred[:, 6] = (raw[:, 9] + 1) / 2 * (Q99[9] - Q01[9]) + Q01[9]  # gripper: decoded, not scored
            row, _ = trajectory_scores(pred, state, gripper)
            rows.append({"model": name, "clip": clip, **row})
        print(f"{name}: {sum(r['model'] == name for r in rows)} clips recovered", flush=True)
    write_rows(os.path.join(args.gen, "metrics_idm_cosmos3.csv"), rows, args.models)


if __name__ == "__main__":
    main()
