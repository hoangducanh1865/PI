"""
Build the two reference "models" for the chosen clips: the real video (ceiling) and the first frame held still for
16 frames (floor), with IRASim's own clip indexing and frame preprocessing, plus the robot state of every clip.

Usage (from video_eval/, with IRASim's packages on PYTHONPATH for decord):
    python scripts/make_references.py --data <DATA_DIR> --out <OUT_DIR>

Clips follow IRASim's Dataset_3D: start frames 0, 16, 32, ... of each episode in annotation_subset/test, 16
consecutive frames, kept only if complete; frames resized to 256 x 320 with torchvision's antialiased resize (its
Resize_Preprocess; the real videos are 480 x 640). Writes <OUT_DIR>/real/<clip>.mp4, <OUT_DIR>/still/<clip>.mp4 and
<OUT_DIR>/states/<clip>.npz (state: 16 x 6 end-effector position and orientation, gripper: 16 continuous gripper
state, instruction), with <clip> = <episode>_<camera>_<start frame> as in IRASim's output names.

Author: Duc Anh Hoang
"""

import argparse
import glob
import json
import os

import decord
import imageio
import numpy as np
import torch
import torchvision.transforms.functional as F

FRAMES = 16
SIZE = (256, 320)
FPS = 4  # IRASim writes its videos at 4 fps


def save(path: str, frames: np.ndarray) -> None:
    with imageio.get_writer(path, fps=FPS) as writer:
        for frame in frames:
            writer.append_data(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    bridge = os.path.join(args.data, "opensource_robotdata", "bridge")
    for kind in ("real", "still", "states"):
        os.makedirs(os.path.join(args.out, kind), exist_ok=True)

    clips = 0
    for path in sorted(glob.glob(os.path.join(bridge, "annotation_subset", "test", "*.json"))):
        with open(path) as f:
            label = json.load(f)
        reader = decord.VideoReader(os.path.join(bridge, label["videos"][0]["video_path"]))
        state, gripper = np.array(label["state"]), np.array(label["continuous_gripper_state"])
        for start in range(0, len(state) - FRAMES + 1, FRAMES):
            ids = list(range(start, start + FRAMES))
            video = torch.from_numpy(reader.get_batch(ids).asnumpy()).permute(0, 3, 1, 2)
            video = torch.stack([F.resize(frame, SIZE, antialias=True) for frame in video])
            video = video.permute(0, 2, 3, 1).numpy()
            name = f"{label['episode_id']}_0_{start}"
            save(os.path.join(args.out, "real", f"{name}.mp4"), video)
            save(os.path.join(args.out, "still", f"{name}.mp4"), np.repeat(video[:1], FRAMES, axis=0))
            np.savez(os.path.join(args.out, "states", f"{name}.npz"), state=state[ids, :6], gripper=gripper[ids],
                     instruction=label["texts"][0])  # fmt: skip
            clips += 1
    print(f"{clips} clips -> {args.out}/{{real,still,states}}")


if __name__ == "__main__":
    main()
