"""
Keep only the low-dimensional trajectories of MimicGen datasets (human source demos and generated demos), dropping the
camera images that make up most of the 1 to 10 GB files, for the human-vs-generated comparison (experiment E1).

Usage (from sim2real_worldmodel/):
    python scripts/extract_mimicgen.py --hdf5 <FILE.hdf5> [<FILE.hdf5> ...] --out <DIR>

Input: MimicGen release files (Hugging Face amandlek/mimicgen_datasets, source/<task>.hdf5 and core/<task>_d<k>.hdf5;
robomimic layout data/demo_<i>/...). Writes <DIR>/<file name>.npz with, per demo i: demo_<i>/eef_pos (T x 3, m),
demo_<i>/eef_quat (T x 4), demo_<i>/gripper_qpos (T x 2), demo_<i>/actions (T x 7), and every per-step array under
datagen_info/ when the file has it (MimicGen's generation records, e.g. which source segment a step comes from); plus
env_args (JSON string). Control rate: 20 Hz (env_args).

Author: Duc Anh Hoang
"""

import argparse
import os

import h5py
import numpy as np

OBS = {"eef_pos": "robot0_eef_pos", "eef_quat": "robot0_eef_quat", "gripper_qpos": "robot0_gripper_qpos"}


def all_datasets(group: h5py.Group):
    """(path, dataset) of every dataset under a group, at any depth."""
    found = []
    group.visititems(lambda key, item: found.append((key, item)) if isinstance(item, h5py.Dataset) else None)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for path in args.hdf5:
        arrays = {}
        with h5py.File(path, "r") as f:
            arrays["env_args"] = np.array(f["data"].attrs["env_args"])
            for name, demo in f["data"].items():
                for key, obs in OBS.items():
                    arrays[f"{name}/{key}"] = demo["obs"][obs][()]
                arrays[f"{name}/actions"] = demo["actions"][()]
                if "datagen_info" in demo:
                    for key, item in all_datasets(demo["datagen_info"]):
                        arrays[f"{name}/datagen_info/{key}"] = item[()]
        out = os.path.join(args.out, os.path.basename(path).replace(".hdf5", ".npz"))
        np.savez_compressed(out, **arrays)
        demos = sum(k.endswith("/eef_pos") for k in arrays)
        extra = sorted({k.split("/datagen_info/")[1] for k in arrays if "/datagen_info/" in k})
        print(f"{out}: {demos} demos, {os.path.getsize(out) / 1e6:.1f} MB, datagen_info: {extra or 'none'}")


if __name__ == "__main__":
    main()
