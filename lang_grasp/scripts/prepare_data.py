"""
Pack the Grasp-Anything++ samples of the authors' Seen / Unseen split into three HDF5 files.

Usage (on Colab, CPU runtime is enough; about 30-40 min, most of it one sequential pass over the 65 GB image zip):
    python scripts/prepare_data.py --raw <RAW_DIR> --out <OUT_DIR> [--splits <DIR_WITH_SEEN_UNSEEN_OBJ>]
    e.g. python scripts/prepare_data.py --raw /content/drive/MyDrive/it/fpt/data/datasets/raw \\
             --out /content/drive/MyDrive/it/fpt/data/datasets/processed/Grasp-Anything-pp

<RAW_DIR> holds Grasp-Anything/image_part_{aa,ab} and Grasp-Anything-pp/*.zip as downloaded from Hugging Face.
Protocol (as in the authors' code, github.com/Fsoft-AIC/Grasp-Anything): the objects of seen.obj are the Base pool,
sorted and cut 90% train / 10% Seen test (train_network.py --split 0.9); the objects of unseen.obj are the Unseen
test. One prompt (object, part) = one sample. Without --splits the two files are downloaded from that repo.

Output in <OUT_DIR>: train.h5, test_seen.h5, test_unseen.h5 and stats.json. Each .h5 holds
    images        uint8 [n_images, 224, 224, 3]    image_ids  str [n_images] (hash = file name in the image zip)
    keys          str [n] (<image>_<object>_<part>) prompts    str [n]         image_index  int32 [n]
    grasps        float32 [n_grasps, 6]: quality, x, y, w, h, theta (degrees); x, y, w, h in pixels of 224 x 224
    grasp_offsets int64 [n + 1]: the grasps of sample i are grasps[grasp_offsets[i]:grasp_offsets[i + 1]]
    masks         uint8 [n, 6272]: part mask at 224 x 224, np.packbits (np.unpackbits(...).reshape(224, 224))
    part_distinct bool [n]: the parts of this object have different grasps and masks, each inside its own part

Author: Duc Anh Hoang
"""

import argparse
import collections
import io
import json
import math
import os
import pickle
import shutil
import tempfile
import time
import urllib.request
import zipfile

import h5py
import numpy as np
import torch
from PIL import Image

SIZE = 224  # network input
RAW_SIZE = 416  # Grasp-Anything images
SPLIT_URL = "https://raw.githubusercontent.com/Fsoft-AIC/Grasp-Anything/d7755f43c5518bd6590b25021054f862e65bddd5/split/grasp-anything/{}.obj"
NEAR_PX = 5  # grasps of two parts closer than this (pixels of 416 x 416) count as the same


class SequentialFile(io.RawIOBase):
    """One file made of the parts of a split zip. A short forward seek reads and drops the bytes instead of jumping:
    on Google Drive a jump costs about 2 s, while sequential reading runs at about 40 MB/s."""

    def __init__(self, paths, skip_limit=256 << 20):
        self.files = [open(p, "rb", buffering=0) for p in paths]
        self.sizes = [os.path.getsize(p) for p in paths]
        self.pos, self.skip_limit = 0, skip_limit
        self.scratch = bytearray(8 << 20)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        target = {0: offset, 1: self.pos + offset, 2: sum(self.sizes) + offset}[whence]
        gap = target - self.pos
        while 0 < gap <= self.skip_limit:
            n = self.readinto(memoryview(self.scratch)[: min(gap, len(self.scratch))])
            if not n:
                break
            gap -= n
        self.pos = target
        return self.pos

    def readinto(self, buf):
        start, n = self.pos, 0
        for f, size in zip(self.files, self.sizes, strict=True):
            if start < size and n < len(buf):
                f.seek(start)
                n += f.readinto(memoryview(buf)[n : n + size - start])
                start = size
            start = max(0, start - size)
        self.pos += n
        return n

    def close(self):
        for f in self.files:
            f.close()
        super().close()


def open_zip(*paths):
    return zipfile.ZipFile(io.BufferedReader(SequentialFile(paths), 8 << 20))


def read_in_file_order(zf, names, decode, every=5000):
    """Read the named entries in the order they are stored (so the reads stay sequential)."""
    infos = sorted((zf.getinfo(n) for n in names), key=lambda info: info.header_offset)
    out, start = {}, time.time()
    for i, info in enumerate(infos, 1):
        out[info.filename] = decode(zf.read(info))
        if i % every == 0 or i == len(infos):
            print(f"  {i}/{len(infos)} read, {time.time() - start:.0f} s", flush=True)
    return out


def load_split(name, splits_dir):
    if splits_dir:
        with open(os.path.join(splits_dir, f"{name}.obj"), "rb") as f:
            return set(pickle.load(f))
    with urllib.request.urlopen(SPLIT_URL.format(name)) as r:
        return set(pickle.loads(r.read()))


def part_distinct(grasps, masks):
    """True if the parts of one object have different grasps and masks and the grasps of each part lie in that part."""
    if len(grasps) < 2:
        return False
    ref = grasps[0]
    near = all(g.shape == ref.shape and np.abs(g[:, 1:3] - ref[:, 1:3]).max() < NEAR_PX for g in grasps)
    same_mask = all(np.array_equal(m, masks[0]) for m in masks)

    def inside(g, m):  # share of grasp centres inside the mask (mask at SIZE, grasps at RAW_SIZE)
        if not len(g):
            return 0.0
        xy = np.clip((g[:, 1:3] * SIZE / RAW_SIZE).round().astype(int), 0, SIZE - 1)
        return m[xy[:, 1], xy[:, 0]].mean()

    return not near and not same_mask and all(inside(g, m) >= 0.5 for g, m in zip(grasps, masks, strict=True))


def write_split(path, keys, prompts, grasps, masks, distinct, images):
    image_ids = sorted({k.split("_")[0] for k in keys})
    index = {h: i for i, h in enumerate(image_ids)}
    scale = np.array([1, SIZE / RAW_SIZE, SIZE / RAW_SIZE, SIZE / RAW_SIZE, SIZE / RAW_SIZE, 1], dtype=np.float32)
    counts = [len(grasps[k]) for k in keys]
    text = h5py.string_dtype()
    with h5py.File(path, "w") as f:
        f.attrs["grasp_columns"] = "quality, x, y, w, h, theta_deg; x, y, w, h in pixels of the 224 x 224 image"
        f.attrs["source"] = "Grasp-Anything++ (airvlab), authors' seen.obj / unseen.obj split"
        f.create_dataset("images", data=np.stack([images[h] for h in image_ids]), chunks=(1, SIZE, SIZE, 3))
        f.create_dataset("image_ids", data=image_ids, dtype=text)
        f.create_dataset("keys", data=keys, dtype=text)
        f.create_dataset("prompts", data=[prompts[k] for k in keys], dtype=text)
        f.create_dataset("image_index", data=np.array([index[k.split("_")[0]] for k in keys], dtype=np.int32))
        f.create_dataset("grasps", data=np.concatenate([grasps[k] * scale for k in keys]).astype(np.float32))
        f.create_dataset("grasp_offsets", data=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))
        f.create_dataset("masks", data=np.stack([masks[k] for k in keys]), compression="gzip")
        f.create_dataset("part_distinct", data=np.array([distinct[k.rsplit("_", 1)[0]] for k in keys]))
    return {
        "samples": len(keys),
        "objects": len({k.rsplit("_", 1)[0] for k in keys}),
        "images": len(image_ids),
        "grasps": int(sum(counts)),
        "part_distinct_samples": int(sum(distinct[k.rsplit("_", 1)[0]] for k in keys)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, help="folder with Grasp-Anything/ and Grasp-Anything-pp/")
    parser.add_argument("--out", required=True, help="output folder (created)")
    parser.add_argument("--splits", help="folder with seen.obj and unseen.obj (default: download them)")
    args = parser.parse_args(argv)
    start = time.time()
    seen, unseen = load_split("seen", args.splits), load_split("unseen", args.splits)
    print(f"split files: {len(seen)} seen objects, {len(unseen)} unseen, {len(seen & unseen)} in both", flush=True)
    pp = os.path.join(args.raw, "Grasp-Anything-pp")

    # Each zip has 4.4M entries; its index takes a few GB of RAM, so one zip is open at a time.
    print("labels:", flush=True)
    wanted = seen | unseen
    with open_zip(os.path.join(pp, "grasp_label_positive.zip")) as z:
        names = [n for n in z.namelist() if n.endswith(".pt") and n[21:-3].rsplit("_", 1)[0] in wanted]
        grasps = read_in_file_order(z, names, lambda b: torch.load(io.BytesIO(b)).numpy().reshape(-1, 6))
    grasps = {n[21:-3]: g for n, g in grasps.items()}  # strip "grasp_label_positive/" and ".pt"
    keys = sorted(grasps)
    print("prompts:", flush=True)
    with open_zip(os.path.join(pp, "grasp_instructions.zip")) as z:
        prompts = read_in_file_order(z, [f"grasp_instructions/{k}.pkl" for k in keys], pickle.loads)
    prompts = {n[19:-4]: p for n, p in prompts.items()}
    print("part masks:", flush=True)

    def small_mask(b):
        mask = Image.fromarray(np.load(io.BytesIO(b)).astype(np.uint8) * 255).resize(
            (SIZE, SIZE), Image.Resampling.NEAREST
        )
        return np.packbits(np.asarray(mask) > 0)

    with open_zip(os.path.join(pp, "part_mask.zip")) as z:
        masks = read_in_file_order(z, [f"part_mask/{k}.npy" for k in keys], small_mask)
    masks = {n[10:-4]: m for n, m in masks.items()}

    by_object = collections.defaultdict(list)
    for k in keys:
        by_object[k.rsplit("_", 1)[0]].append(k)
    distinct = {
        o: part_distinct([grasps[k] for k in ks], [np.unpackbits(masks[k]).reshape(SIZE, SIZE) for k in ks])
        for o, ks in by_object.items()
    }

    print("images (one sequential pass over the image zip):", flush=True)
    raw_images = os.path.join(args.raw, "Grasp-Anything")
    with open_zip(os.path.join(raw_images, "image_part_aa"), os.path.join(raw_images, "image_part_ab")) as z:

        def small_image(b):
            return np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR))

        hashes = sorted({k.split("_")[0] for k in keys})
        images = read_in_file_order(z, [f"image/{h}.jpg" for h in hashes], small_image, every=1000)
    images = {n[6:-4]: im for n, im in images.items()}

    seen_keys = [k for k in keys if k.rsplit("_", 1)[0] in seen]
    cut = math.floor(0.9 * len(seen_keys))  # the authors' train_network.py: --split 0.9 on the sorted list
    splits = {
        "train": seen_keys[:cut],
        "test_seen": seen_keys[cut:],
        "test_unseen": [k for k in keys if k.rsplit("_", 1)[0] in unseen],
    }
    os.makedirs(args.out, exist_ok=True)
    stats = {}
    with tempfile.TemporaryDirectory() as tmp:  # write on the local disk, then copy (Drive is slow at small writes)
        for name, split_keys in splits.items():
            stats[name] = write_split(
                os.path.join(tmp, f"{name}.h5"), split_keys, prompts, grasps, masks, distinct, images
            )
            shutil.copy(os.path.join(tmp, f"{name}.h5"), os.path.join(args.out, f"{name}.h5"))
            print(f"{name}: {stats[name]}", flush=True)
    stats["split_objects_without_labels"] = len((seen | unseen) - set(by_object))
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(f"done in {(time.time() - start) / 60:.1f} min: {args.out}", flush=True)


if __name__ == "__main__":
    main()
