"""
Sort the failures of an evaluation (src/eval.py) into types, to see what a model gets wrong.

Usage (any platform; CPU is enough, a few minutes):
    python scripts/analyze_errors.py --eval <EVAL_RUN_DIR> --data <PROCESSED_DIR> --out <OUT_DIR> [--draw 8]

For every failed prompt, with the predicted centre (x, y) and the masks of the objects in the image (from all three
splits, since an image of one split can hold objects of another):
    no_grasp        Q has no peak above 0.2, nothing is predicted
    wrong_object    the centre lies on another annotated object, not on the one in the prompt
    background      the centre lies on no annotated object
    wrong_angle     the centre is on the right object and some ground-truth grasp would match (IoU > 25%) if the
                    angle were ignored, but the angle is off by 30 degrees or more
    wrong_geometry  the centre is on the right object, but position or opening miss every ground-truth grasp
Writes into <OUT_DIR>: summary.json (shares per split, also for part-distinct objects), failures.csv (one row per
failed prompt) and failures_<type>.png (--draw random failures per type: prompt, target mask in green, other objects
in blue, ground truth in red, prediction in yellow). --export "background=100,wrong_object=50" also writes random
failures of those types to <OUT_DIR>/<--export-name>/ (default export): one 448 x 448 image each with only the
predicted rectangle and its centre in yellow (no mask, no ground truth, so a judge is not biased) and manifest.csv,
the input of scripts/judge_failures.py. --exclude <manifest.csv> skips the prompts of an earlier export.

Author: Duc Anh Hoang
"""

import argparse
import collections
import csv
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # task root, for src.*
sys.dont_write_bytecode = True
from src.utils.grasp import corners, iou  # noqa: E402

SPLITS = ("train", "test_seen", "test_unseen")
TYPES = ("no_grasp", "wrong_object", "background", "wrong_angle", "wrong_geometry")


def object_of(key):
    return key.rsplit("_", 1)[0]  # <image>_<object>_<part> -> <image>_<object>


def object_masks(data_dir, images):
    """{image: {object: mask (224 x 224 bool, union of its parts)}} for the given images, from all splits."""
    masks = collections.defaultdict(dict)
    for split in SPLITS:
        with h5py.File(os.path.join(data_dir, f"{split}.h5")) as f:
            keys = [k.decode() for k in f["keys"][:]]
            rows = [i for i, k in enumerate(keys) if k.split("_")[0] in images]
            packed = f["masks"][:] if rows else None
        for i in rows:
            m = np.unpackbits(packed[i]).reshape(224, 224).astype(bool)
            image, obj = keys[i].split("_")[0], object_of(keys[i])
            masks[image][obj] = masks[image][obj] | m if obj in masks[image] else m
    return masks


def classify(pred, gts, target, others):
    if np.isnan(pred[0]):
        return "no_grasp"
    x, y = (int(np.clip(round(v), 0, 223)) for v in pred[:2])
    if not target[y, x]:
        return "wrong_object" if any(m[y, x] for m in others) else "background"
    if any(iou(pred, g[1:], max_angle_deg=180.0) > 0.25 for g in gts):
        return "wrong_angle"
    return "wrong_geometry"


def draw(image, prompt, target, others, gts, pred, kind):
    rgb = image.copy()
    for m in others:
        rgb[m] = (0.6 * rgb[m] + [0, 0, 100]).astype(np.uint8)
    rgb[target] = (0.5 * rgb[target] + [0, 127, 0]).astype(np.uint8)
    img = Image.fromarray(rgb).resize((448, 448))
    d = ImageDraw.Draw(img)
    for g in gts[:6]:
        d.polygon([tuple(2 * p) for p in corners(*g[1:])], outline=(255, 0, 0))
    if not np.isnan(pred[0]):
        pts = 2 * corners(*pred)
        d.polygon([tuple(p) for p in pts], outline=(255, 255, 0), width=3)
    panel = Image.new("RGB", (448, 478), "white")
    panel.paste(img, (0, 30))
    ImageDraw.Draw(panel).text((4, 8), f"[{kind}] {prompt}", fill="black")
    return panel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", required=True, help="run folder of src/eval.py (predictions_<split>.npz)")
    parser.add_argument("--data", required=True, help="folder with the packed splits (scripts/prepare_data.py)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--draw", type=int, default=8, help="failures drawn per type")
    parser.add_argument(
        "--export", default="", help='failures to export per type, e.g. "background=100,wrong_object=50"'
    )
    parser.add_argument("--export-name", default="export", help="folder of the export, inside --out")
    parser.add_argument("--exclude", help="manifest.csv of an earlier export: do not export those prompts again")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    summary, rows, to_draw, loaded = {}, [], collections.defaultdict(list), {}
    for split in ("test_seen", "test_unseen"):
        pred = np.load(os.path.join(args.eval, f"predictions_{split}.npz"))
        with h5py.File(os.path.join(args.data, f"{split}.h5")) as f:
            keys = [k.decode() for k in f["keys"][:]]
            prompts = [p.decode() for p in f["prompts"][:]]
            grasps, offsets, distinct = f["grasps"][:], f["grasp_offsets"][:], f["part_distinct"][:]
        if list(pred["keys"]) != keys:
            raise ValueError(f"predictions_{split}.npz does not match {split}.h5")
        failed = np.flatnonzero(~pred["success"])
        masks = object_masks(args.data, {keys[i].split("_")[0] for i in failed})
        loaded[split] = (pred["grasps"], masks, keys)
        counts = collections.Counter()
        distinct_counts = collections.Counter()
        for i in failed:
            image, obj = keys[i].split("_")[0], object_of(keys[i])
            target = masks[image][obj]
            others = [m for o, m in masks[image].items() if o != obj]
            gts = grasps[offsets[i] : offsets[i + 1]]
            kind = classify(pred["grasps"][i], gts, target, others)
            counts[kind] += 1
            distinct_counts[kind] += bool(distinct[i])
            rows.append([split, keys[i], prompts[i], kind])
            to_draw[kind].append((split, i))
        n, n_distinct = len(keys), int(distinct.sum())
        summary[split] = {
            "prompts": n,
            "success": round(float(pred["success"].mean()), 4),
            "failures": {k: round(counts[k] / n, 4) for k in TYPES},
            "failures_share_of_failed": {k: round(counts[k] / max(len(failed), 1), 4) for k in TYPES},
            "part_distinct_prompts": n_distinct,
            "part_distinct_success": round(float(pred["success"][distinct].mean()), 4) if n_distinct else None,
            "other_prompts_success": round(float(pred["success"][~distinct].mean()), 4),
        }
        print(split, json.dumps(summary[split], indent=1), flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(args.out, "failures.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["split", "key", "prompt", "type"], *rows])

    for kind, items in to_draw.items():  # drawings of random failures per type
        panels = []
        for j in rng.choice(len(items), min(args.draw, len(items)), replace=False):
            split, i = items[j]
            pred_grasps, split_masks, _ = loaded[split]
            with h5py.File(os.path.join(args.data, f"{split}.h5")) as f:
                key, prompt = f["keys"][i].decode(), f["prompts"][i].decode()
                image = f["images"][f["image_index"][i]]
                gts = f["grasps"][f["grasp_offsets"][i] : f["grasp_offsets"][i + 1]]
            masks = split_masks[key.split("_")[0]]
            target = masks[object_of(key)]
            others = [m for o, m in masks.items() if o != object_of(key)]
            panels.append(draw(image, prompt, target, others, gts, pred_grasps[i], kind))
        if not panels:  # --draw 0
            continue
        sheet = Image.new("RGB", (448 * 4, 478 * ((len(panels) + 3) // 4)), "white")
        for k, p in enumerate(panels):
            sheet.paste(p, ((k % 4) * 448, (k // 4) * 478))
        sheet.save(os.path.join(args.out, f"failures_{kind}.png"))

    if args.export:  # clean images for an outside judge: the prediction only
        export = os.path.join(args.out, args.export_name)
        os.makedirs(export, exist_ok=True)
        manifest = [["file", "split", "key", "prompt", "type"]]
        done = set()
        if args.exclude:
            with open(args.exclude) as f:
                done = {(r["split"], r["key"]) for r in csv.DictReader(f)}
        for item in args.export.split(","):
            kind, n = item.split("=")
            items = [(split, i) for split, i in to_draw[kind] if (split, loaded[split][2][i]) not in done]
            for j in rng.choice(len(items), min(int(n), len(items)), replace=False):
                split, i = items[j]
                with h5py.File(os.path.join(args.data, f"{split}.h5")) as f:
                    key, prompt = f["keys"][i].decode(), f["prompts"][i].decode()
                    image = Image.fromarray(f["images"][f["image_index"][i]]).resize((448, 448))
                pred = loaded[split][0][i]
                d = ImageDraw.Draw(image)
                d.polygon([tuple(p) for p in 2 * corners(*pred)], outline=(255, 255, 0), width=3)
                x, y = 2 * pred[0], 2 * pred[1]
                d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 0), outline=(0, 0, 0))
                name = f"{kind}_{len(manifest):03d}.png"
                image.save(os.path.join(export, name))
                manifest.append([name, split, key, prompt, kind])
        with open(os.path.join(export, "manifest.csv"), "w", newline="") as f:
            csv.writer(f).writerows(manifest)
    print(f"written: {args.out}", flush=True)


if __name__ == "__main__":
    main()
