"""
Generate the chosen Bridge clips with NVIDIA Cosmos-Predict2.5-2B, through the official repo's own entry points, and
store them like the IRASim clips (16 frames, 256 x 320, 4 fps) so every metric script applies unchanged.

Usage (from video_eval/, any Python with imageio and Pillow, on an Ampere-or-newer GPU with the repo installed by
scripts/setup_cosmos.sh, HF_TOKEN set; the repo's own entry points run with the repo's .venv):
    python scripts/run_cosmos.py --cosmos <REPO> --data <DATA_DIR> --gen <GEN_DIR> --variant action
        [--annotation annotation_<NAME> --name cosmos_action_<NAME>]
    python scripts/run_cosmos.py --cosmos <REPO> --data <DATA_DIR> --gen <GEN_DIR> --variant base

Variants:
- action: robot/action-cond (post-trained on Bridge; actions = the same 7-D relative end-effector motion x 20 as IRASim,
  computed by the repo's own loader from our annotation JSONs, so corrupted annotations work too). One JSON per clip,
  states cut at the clip's last frame; the repo's start_frame_idx is global, so clips are run in groups by start frame.
  The repo generates chunks of 12 actions from 1 conditioning frame (guidance 0, 35 steps, 256 x 320, as its Bridge
  example); chunk 1 gives frames 0 to 12, chunk 2 (3 actions, zero-padded) frames 13 to 24; frames 0 to 15 are kept.
- base: base/post-trained, image + text (no actions): the clip's first frame (original 480 x 640) and its instruction,
  49 frames at the model's 16 fps (about 3 s, the length of a 16-frame Bridge clip at 5 Hz), 480 x 640, guidance 7,
  guardrails off; every third frame (0, 3, ..., 45) is kept and resized to 256 x 320. All clips not yet generated go
  to one call of the repo's inference.py, so the model loads once (about 1.5 min) instead of once per clip.
Writes <GEN_DIR>/<NAME>/seed0/<clip>.mp4 (<NAME> default cosmos_action or cosmos_base) and the repo logs under
--work. The repo fixes the noise seed itself (action: the chunk index; base: seed 0).

Author: Duc Anh Hoang
"""

import argparse
import glob
import json
import os
import subprocess

import imageio.v2 as imageio
import numpy as np
from PIL import Image

FRAMES = 16
SIZE = (320, 256)  # PIL (width, height)
FPS = 4


def save(path: str, frames: list[np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = [np.array(Image.fromarray(f).resize(SIZE, Image.BICUBIC)) for f in frames]
    with imageio.get_writer(path, fps=FPS) as writer:
        for frame in frames:
            writer.append_data(frame)


def run(command: list[str], cwd: str, log: str) -> None:
    with open(log, "w") as f:
        code = subprocess.run(command, cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode
    if code:
        raise SystemExit(f"{' '.join(command[:3])} failed (exit {code}), see {log}")


def action_variant(args, clips: list[str], bridge: str, out: str) -> None:
    starts = sorted({int(c.rsplit("_", 1)[1]) for c in clips})
    for start in starts:
        group = [c for c in clips if int(c.rsplit("_", 1)[1]) == start]
        root = os.path.join(args.work, f"start{start}")
        folder = os.path.join(root, "annotation", "test")
        os.makedirs(folder, exist_ok=True)
        if not os.path.exists(os.path.join(root, "videos")):
            os.symlink(os.path.join(bridge, "videos"), os.path.join(root, "videos"))
        for clip in group:
            with open(os.path.join(bridge, args.annotation, "test", f"{clip.split('_')[0]}.json")) as f:
                label = json.load(f)
            for key in ("state", "continuous_gripper_state"):
                label[key] = label[key][: start + FRAMES]
            with open(os.path.join(folder, f"{clip}.json"), "w") as f:
                json.dump(label, f)
        params = {
            "name": f"start{start}", "input_root": root, "input_json_sub_folder": "annotation/test",
            "save_root": os.path.join(root, "output"), "guidance": 0, "resolution": "256,320", "camera_id": 0,
            "start": 0, "end": len(group), "start_frame_idx": start, "save_fps": 20, "num_latent_conditional_frames": 1,
            "action_scaler": 20.0, "gripper_scale": 1.0, "chunk_size": 12,
        }  # fmt: skip
        with open(os.path.join(root, "params.json"), "w") as f:
            json.dump(params, f, indent=1)
        command = [args.python, "examples/action_conditioned.py", "-i", os.path.join(root, "params.json"),
                   "--output-dir", os.path.join(root, "setup")]  # fmt: skip
        run(command, args.cosmos, os.path.join(root, "run.log"))
        for clip in group:
            frames = list(imageio.mimread(os.path.join(root, "output", f"{clip}_chunk.mp4"), memtest=False))
            save(os.path.join(out, f"{clip}.mp4"), frames[:FRAMES])
        print(f"start frame {start}: {len(group)} clips", flush=True)


def base_variant(args, clips: list[str], bridge: str, out: str) -> None:
    os.makedirs(args.work, exist_ok=True)
    todo = [c for c in clips if not os.path.exists(os.path.join(out, f"{c}.mp4"))]
    inputs = []
    for clip in todo:
        episode, _, start = clip.split("_")
        video = imageio.get_reader(os.path.join(bridge, "videos", "test", episode, "rgb.mp4"))
        image = os.path.join(args.work, f"{clip}.png")
        imageio.imwrite(image, video.get_data(int(start)))
        instruction = str(np.load(os.path.join(args.gen, "states", f"{clip}.npz"))["instruction"])
        params = {
            "inference_type": "image2world", "name": clip, "input_path": image,
            "prompt": f"A robot arm in a toy kitchen performs the task: {instruction}. Realistic, steady camera.",
            "num_output_frames": 49, "resolution": "480,640", "guidance": 7, "seed": 0,
        }  # fmt: skip
        inputs.append(os.path.join(args.work, f"{clip}.json"))
        with open(inputs[-1], "w") as f:
            json.dump(params, f, indent=1)
    if not todo:
        return
    result = os.path.join(args.work, "output")
    command = [args.python, "examples/inference.py", "-i", *inputs, "--output-dir", result,
               "--model", "2B/post-trained", "--disable-guardrails"]  # fmt: skip
    run(command, args.cosmos, os.path.join(args.work, "run.log"))  # one model load for all clips
    for clip in todo:
        frames = list(imageio.mimread(os.path.join(result, f"{clip}.mp4"), memtest=False))
        save(os.path.join(out, f"{clip}.mp4"), frames[: 3 * FRAMES : 3])
    print(f"base: {len(todo)} clips", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cosmos", required=True, help="clone of nvidia-cosmos/cosmos-predict2.5, installed")
    parser.add_argument("--data", required=True, help="folder holding opensource_robotdata/ (scripts/fetch_irasim.py)")
    parser.add_argument("--gen", required=True, help="folder with real/ and states/ (scripts/make_references.py)")
    parser.add_argument("--variant", choices=("action", "base"), required=True)
    parser.add_argument("--annotation", default="annotation_subset", help="action: annotation folder of the actions")
    parser.add_argument("--name", help="output folder name (default cosmos_<variant>)")
    parser.add_argument("--work", default="/tmp/cosmos_work")
    parser.add_argument("--clips", nargs="+", help="only these clips (default: every clip in <GEN_DIR>/real)")
    args = parser.parse_args()
    args.python = os.path.join(args.cosmos, ".venv", "bin", "python")
    bridge = os.path.join(args.data, "opensource_robotdata", "bridge")
    clips = args.clips or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(args.gen, "real", "*.mp4")))
    out = os.path.join(args.gen, args.name or f"cosmos_{args.variant}", "seed0")
    args.work = os.path.join(args.work, args.name or f"cosmos_{args.variant}")
    (action_variant if args.variant == "action" else base_variant)(args, clips, bridge, out)


if __name__ == "__main__":
    main()
