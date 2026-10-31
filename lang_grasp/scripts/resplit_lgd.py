"""
Re-split the packed Grasp-Anything++ data (scripts/prepare_data.py) into the official split of the LGD code.

Usage (any platform, CPU, a few minutes):
    python scripts/resplit_lgd.py --src <PROCESSED_DIR> --out <OUT_DIR>
    e.g. python scripts/resplit_lgd.py --src /content/drive/MyDrive/it/fpt/data/datasets/processed/Grasp-Anything-pp \\
             --out /content/work/Grasp-Anything-pp-lgd

The LGD repository (github.com/Fsoft-AIC/LGD, the code of the Grasp-Anything++ paper) lists the objects of its split in
split/grasp-anything++/{train/seen,test/seen,test/unseen}.obj (14,516 / 573 / 230 objects; pickled lists of
"<image>_<object>"). Every one of them that has Grasp-Anything++ labels is already in <PROCESSED_DIR> (checked on
2026-10-27: 10,130 + 515 + 212 objects), so the new split is made from the packed files, without the raw data.
Output in <OUT_DIR>, same format as prepare_data.py:
    train.h5             official train objects
    test_seen.h5         official Seen test objects
    test_unseen.h5       official Unseen test objects
    test_unseen_all.h5   every labelled object of the Grasp-Anything unseen list (the larger Unseen test used before)
    stats.json
The pickles are read with an unpickler that only accepts lists of strings.

Author: Duc Anh Hoang
"""

import argparse
import io
import json
import os
import pickle
import shutil
import sys
import urllib.request

import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import write_split  # noqa: E402

LGD_URL = "https://raw.githubusercontent.com/Fsoft-AIC/LGD/0223d43e5ea07d459f21296bb1305e2e94d86cbd/split/grasp-anything++/{}.obj"
SOURCE = "Grasp-Anything++ (airvlab), official split of github.com/Fsoft-AIC/LGD (split/grasp-anything++)"


class StringListUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused to load {module}.{name}")


def load_lgd(name):
    with urllib.request.urlopen(LGD_URL.format(name)) as r:
        objects = StringListUnpickler(io.BytesIO(r.read())).load()
    assert isinstance(objects, list) and all(isinstance(o, str) for o in objects)
    return set(objects)


def read_packed(path):
    with h5py.File(path) as f:
        keys = [k.decode() for k in f["keys"][:]]
        prompts = [p.decode() for p in f["prompts"][:]]
        image_ids = [h.decode() for h in f["image_ids"][:]]
        images, index = f["images"][:], f["image_index"][:]
        grasps, offsets, masks, distinct = f["grasps"][:], f["grasp_offsets"][:], f["masks"][:], f["part_distinct"][:]
    return {
        k: (
            prompts[i],
            grasps[offsets[i] : offsets[i + 1]],
            masks[i],
            bool(distinct[i]),
            image_ids[index[i]],
            images[index[i]],
        )
        for i, k in enumerate(keys)
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="folder of prepare_data.py (train / test_seen / test_unseen .h5)")
    parser.add_argument("--out", required=True, help="output folder (created)")
    args = parser.parse_args(argv)
    samples = {}
    for split in ("train", "test_seen", "test_unseen"):
        samples.update(read_packed(os.path.join(args.src, f"{split}.h5")))
    old_unseen = {k.rsplit("_", 1)[0] for k in read_packed(os.path.join(args.src, "test_unseen.h5"))}
    objects = {
        "train": load_lgd("train/seen"),
        "test_seen": load_lgd("test/seen"),
        "test_unseen": load_lgd("test/unseen"),
        "test_unseen_all": old_unseen,
    }
    prompts = {k: v[0] for k, v in samples.items()}
    grasps = {k: v[1] for k, v in samples.items()}
    masks = {k: v[2] for k, v in samples.items()}
    distinct = {k.rsplit("_", 1)[0]: v[3] for k, v in samples.items()}
    images = {v[4]: v[5] for v in samples.values()}
    os.makedirs(args.out, exist_ok=True)
    stats = {}
    for name, objs in objects.items():
        keys = sorted(k for k in samples if k.rsplit("_", 1)[0] in objs)
        stats[name] = write_split(
            os.path.join(args.out, f"{name}.h5"),
            keys,
            prompts,
            grasps,
            masks,
            distinct,
            images,
            raw_pixels=False,
            source=SOURCE,
        )
        stats[name]["listed_objects"] = len(objs)
        print(f"{name}: {stats[name]}", flush=True)
    train_objects = {k.rsplit("_", 1)[0] for k in samples if k.rsplit("_", 1)[0] in objects["train"]}
    test_objects = objects["test_seen"] | objects["test_unseen"] | objects["test_unseen_all"]
    assert not train_objects & test_objects, "an official train object is also a test object"
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    if os.path.exists(os.path.join(args.src, "stats.json")):
        shutil.copy(os.path.join(args.src, "stats.json"), os.path.join(args.out, "stats_source.json"))
    print(f"done: {args.out}", flush=True)


if __name__ == "__main__":
    main()
