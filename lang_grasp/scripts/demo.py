"""
Visual demo of the pipeline: image + prompt -> grasp and output maps. One PNG per run in <--out>.

Usage (commands with real paths: terminal_commands.md):
    python scripts/demo.py --ckpt <CKPT> --sample <SPLIT>:<INDEX>                    # a dataset sample
    python scripts/demo.py --ckpt <CKPT> --image <FILE> --prompt "<TEXT>"            # any image
    python scripts/demo.py --ckpt <CKPT> --sample <SPLIT>:<INDEX> --prompt "<TEXT>"  # same image, another prompt
    python scripts/demo.py --ckpt <CKPT_A> <CKPT_B> --sample <SPLIT>:<INDEX>         # two models, one row each
    python scripts/demo.py --ckpt <CKPT> --sample <SPLIT>:<INDEX> --gate oracle      # grasp inside the true mask
    python scripts/demo.py --ckpt <CKPT> --sample <SPLIT>:<INDEX> --gate detector    # grasp inside the detector box
    python scripts/demo.py --ckpt <CKPT> --sample <SPLIT>:<INDEX> --regions          # regions of the contrastive loss

--data, --out and --boxes default to $DEMO_DATA, $DEMO_OUT and $DEMO_BOXES. Each row shows the grasp (yellow), the
correct grasps (red) and the four output maps. With a new --prompt the run is not scored: the correct grasps of the
sample belong to its own prompt. --regions shows the regions of the loss; with an RTC-Grasp checkpoint
also the probability that the prompt means each region. A new --prompt is encoded with CLIP (downloaded once).

Author: Duc Anh Hoang
"""

import argparse
import os
import re
import sys

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # task root, for src.*
sys.dont_write_bytecode = True
from src.data.grasp_anything import GraspAnything  # noqa: E402
from src.models.grasp_module import GraspModule  # noqa: E402
from src.utils.grasp import corners, decode, success  # noqa: E402

SIZE, GAP, TOP = 300, 8, 28
FONT = ImageFont.load_default(size=18)
SCALE = SIZE / 224


def encode(prompt, tag, device):
    """CLIP features of a new prompt, as scripts/cache_text_emb.py stores them."""
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    from transformers.utils import logging

    logging.set_verbosity_error()  # CLIP's vision weights are not used here
    name = f"openai/{tag}"
    tokens = AutoTokenizer.from_pretrained(name)([prompt], padding=True, truncation=True, return_tensors="pt")
    out = CLIPTextModelWithProjection.from_pretrained(name)(**tokens)
    mask = tokens["attention_mask"].bool()
    return {
        "tokens": out.last_hidden_state.detach().to(device),
        "mask": mask.to(device),
        "emb": out.text_embeds.detach().to(device),
    }


def load_sample(args, tag, device):
    """Everything one run needs: the network input, what to draw, and the mask or box of a gate."""
    if args.image:
        rgb = np.asarray(Image.open(args.image).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
        image = rgb.astype(np.float32) / 255.0
        image = torch.from_numpy((image - image.mean()).transpose(2, 0, 1))[None].to(device)
        return dict(
            rgb=rgb,
            image=image,
            prompt=args.prompt,
            gts=None,
            region=None,
            name=os.path.splitext(os.path.basename(args.image))[0],
            **encode(args.prompt, tag, device),
        )
    split, index = args.sample.split(":")
    i = int(index)
    dataset = GraspAnything(args.data, split, tag, regions=args.regions)
    item = dataset[i]
    with h5py.File(os.path.join(args.data, f"{split}.h5")) as f:
        prompt = f["prompts"][i].decode()
        mask = np.unpackbits(f["masks"][i]).reshape(224, 224).astype(bool)
    sample = dict(
        rgb=dataset.images[dataset.image_index[i]],
        image=item["image"][None].to(device),
        gts=dataset.grasps(i),
        name=f"{split}_{i}",
        prompt=prompt,
    )
    sample.update(
        tokens=item["text_tokens"][None].to(device),
        mask=item["text_mask"][None].to(device),
        emb=item["text_emb"][None].to(device),
    )
    if args.prompt:  # the correct grasps belong to the sample's own prompt, so this run cannot be scored
        sample.update(prompt=args.prompt, gts=None, **encode(args.prompt, tag, device))
    sample["region"] = None
    if args.gate == "oracle":
        sample["region"] = mask
    elif args.gate == "detector":
        box = np.load(os.path.join(args.boxes, f"boxes_{split}.npz"))["boxes"][i]
        if not np.isnan(box).any():
            x0, y0, x1, y1 = np.clip(np.round(box), 0, 224).astype(int)
            sample["region"] = np.zeros((224, 224), dtype=bool)
            sample["region"][y0:y1, x0:x1] = True
    if args.regions:
        sample.update(regions=item["regions"][None].to(device), valid=item["region_valid"][None].to(device))
    return sample


def draw_grasps(sample, grasp):
    rgb = np.array(sample["rgb"], dtype=np.float32)
    if sample["region"] is not None:
        rgb[~sample["region"]] *= 0.35  # the part of the image the gate rules out
    tile = Image.fromarray(rgb.astype(np.uint8)).resize((SIZE, SIZE))
    draw = ImageDraw.Draw(tile)
    for g in [] if sample["gts"] is None else sample["gts"][:6]:
        draw.polygon([tuple(p * SCALE) for p in corners(*g[1:])], outline=(230, 30, 30))
    if grasp is not None:
        draw.polygon([tuple(p * SCALE) for p in corners(*grasp)], outline=(255, 255, 0), width=3)
        x, y = grasp[0] * SCALE, grasp[1] * SCALE
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 0), outline=(0, 0, 0))
    return tile


def gray(values, low, high):
    return (
        Image.fromarray((255 * np.clip((values - low) / (high - low), 0, 1)).astype(np.uint8))
        .resize((SIZE, SIZE))
        .convert("RGB")
    )


def sheet(title, rows):
    """rows: lists of (label, tile); one PNG with the prompt on top."""
    columns = max(len(r) for r in rows)
    canvas = Image.new("RGB", (columns * (SIZE + GAP) - GAP, TOP + len(rows) * (SIZE + 2 * TOP)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), title, fill="black", font=FONT)
    for r, row in enumerate(rows):
        y = TOP + r * (SIZE + 2 * TOP)
        for c, (label, tile) in enumerate(row):
            draw.text((c * (SIZE + GAP) + 2, y + 4), label, fill="black", font=FONT)
            canvas.paste(tile, (c * (SIZE + GAP), y + TOP))
    return canvas


def model_name(module):
    return "RTC-Grasp" if module.hparams.contrast_weight else "GR-ConvNet + CLIP"


@torch.no_grad()
def grasp_row(module, sample):
    maps = module.net(sample["image"], sample["tokens"], sample["mask"]).float().cpu().numpy()[0]
    grasp = decode(*maps, region=sample["region"])
    verdict = "" if sample["gts"] is None else ": success" if success(grasp, sample["gts"]) else ": failure"
    verdict = ": no grasp" if grasp is None else verdict
    print(
        model_name(module)
        + (verdict or ": grasp")
        + ("" if grasp is None else f" at x={grasp[0]:.0f}, y={grasp[1]:.0f}")
    )
    names = [("Q", 0, 1), ("cos 2 theta", -1, 1), ("sin 2 theta", -1, 1), ("width", 0, 1)]
    return [(model_name(module) + verdict, draw_grasps(sample, grasp))] + [
        (n, gray(m, lo, hi)) for (n, lo, hi), m in zip(names, maps, strict=True)
    ]


@torch.no_grad()
def region_row(module, sample):
    side = sample["regions"].shape[-1]
    regions, valid = sample["regions"][0].cpu().numpy(), sample["valid"][0].cpu().numpy()
    prob = None
    if module.contrast is not None:
        _, features = module.net(sample["image"], sample["tokens"], sample["mask"], return_features=True)
        prob = (
            module.contrast.scores(features, sample["regions"], sample["valid"], sample["emb"])
            .softmax(1)[0]
            .cpu()
            .numpy()
        )
    row = [("Image", Image.fromarray(np.array(sample["rgb"])).resize((SIZE, SIZE)))]
    last = len(valid) - 1
    for k in np.flatnonzero(valid):
        label = "Positive" if k == 0 else "Background" if k == last else "Other object"
        mask = np.kron(regions[k] > 0.5, np.ones((224 // side, 224 // side), dtype=bool))
        tint = np.array(sample["rgb"], dtype=np.float32) * 0.25 + 190
        tint[mask] = np.array(sample["rgb"], dtype=np.float32)[mask]
        tile = Image.fromarray(tint.clip(0, 255).astype(np.uint8)).resize((SIZE, SIZE))
        text = label if prob is None else f"{label}: {prob[k]:.0%}"
        print(text)
        row.append((text, tile))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", nargs="+", required=True, help="one checkpoint, or two to compare them")
    parser.add_argument("--sample", help="<split>:<index>, e.g. test_unseen:14")
    parser.add_argument("--image", help="any image file, with --prompt")
    parser.add_argument("--prompt", help="replaces the prompt of the sample")
    parser.add_argument("--gate", choices=("oracle", "detector"), help="restrict the grasp to the mask or the box")
    parser.add_argument("--regions", action="store_true", help="show the regions of the contrastive loss")
    parser.add_argument("--data", default=os.environ.get("DEMO_DATA"), help="folder of the packed splits")
    parser.add_argument("--out", default=os.environ.get("DEMO_OUT"), help="folder of the PNGs")
    parser.add_argument("--boxes", default=os.environ.get("DEMO_BOXES"), help="folder of boxes_<split>.npz")
    args = parser.parse_args(argv)
    if bool(args.sample) == bool(args.image) or (args.image and not args.prompt):
        parser.error("give --sample, or --image with --prompt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    modules = [GraspModule.load_from_checkpoint(c, map_location=device).eval() for c in args.ckpt]
    sample = load_sample(args, modules[0].hparams.text_model, device)
    make_row = region_row if args.regions else grasp_row
    rows = [make_row(m, sample) for m in modules]

    flags = (("regions", args.regions), (args.gate, args.gate), ("compare", len(modules) > 1))
    tags = [tag for tag, used in flags if used]
    if args.prompt:
        tags.append(re.sub(r"\W+", "_", args.prompt.lower()).strip("_")[:40])
    path = os.path.join(args.out, "_".join([sample["name"], *tags]) + ".png")
    os.makedirs(args.out, exist_ok=True)
    sheet(f'Prompt: "{sample["prompt"]}"', rows).save(path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
