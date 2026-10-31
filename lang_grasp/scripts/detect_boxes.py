"""
Find the prompted object with an open-vocabulary detector (Grounding DINO), for the detector gate of src/eval.py.

Usage (a GPU; about 15 min for the three test splits on an A100, an assumption):
    python scripts/detect_boxes.py --data <PROCESSED_DIR> --out <BOXES_DIR> [--splits test_seen test_unseen ...]
        [--model IDEA-Research/grounding-dino-base] [--min-score 0.3] [--batch 32]

For every sample of <PROCESSED_DIR>/<split>.h5 the detector reads the 224 x 224 image and the whole prompt ("lift the
hairbrush by its handle."); the box kept is the query with the highest matching score over the prompt's word tokens.
Writes <BOXES_DIR>/boxes_<split>.npz, in the order of <split>.h5, after each split:
    keys    str [n]
    boxes   float32 [n, 4]  x0, y0, x1, y1 in pixels of the 224 x 224 image; NaN when the best score < --min-score
    scores  float32 [n]

Author: Duc Anh Hoang
"""

import argparse
import os
import time

import h5py
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", nargs="+", default=["test_seen", "test_unseen", "test_unseen_all"])
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
    tok = processor.tokenizer
    special = torch.tensor([tok.cls_token_id, tok.sep_token_id, tok.convert_tokens_to_ids(".")], device=device)
    os.makedirs(args.out, exist_ok=True)
    for split in args.splits:
        start = time.time()
        with h5py.File(os.path.join(args.data, f"{split}.h5")) as f:
            images, index = f["images"][:], f["image_index"][:]
            keys = [k.decode() for k in f["keys"][:]]
            prompts = [p.decode().lower().strip().rstrip(".") + "." for p in f["prompts"][:]]
        side = images.shape[1]
        boxes = np.full((len(keys), 4), np.nan, dtype=np.float32)
        scores = np.zeros(len(keys), dtype=np.float32)
        for b in range(0, len(keys), args.batch):
            ids = range(b, min(b + args.batch, len(keys)))
            inputs = processor(
                images=[Image.fromarray(images[index[i]]) for i in ids],
                text=[prompts[i] for i in ids],
                return_tensors="pt",
                padding=True,
            ).to(device)
            with torch.no_grad():
                out = model(**inputs)
            n_tok = inputs["input_ids"].shape[1]
            words = inputs["attention_mask"].bool() & ~torch.isin(inputs["input_ids"], special)  # [B, T]
            match = out.logits[..., :n_tok].sigmoid().masked_fill(~words[:, None, :], 0.0).amax(-1)  # [B, queries]
            best = match.argmax(-1)
            rows = torch.arange(len(ids), device=device)
            cx, cy, w, h = (out.pred_boxes[rows, best] * side).unbind(-1)
            box = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).float().cpu().numpy()
            score = match[rows, best].float().cpu().numpy()
            scores[b : b + len(ids)] = score
            boxes[b : b + len(ids)] = np.where((score >= args.min_score)[:, None], box, np.nan)
        np.savez(os.path.join(args.out, f"boxes_{split}.npz"), keys=np.array(keys), boxes=boxes, scores=scores)
        found = np.isfinite(boxes[:, 0]).mean()
        print(f"{split}: {len(keys)} prompts, box found for {found:.1%}, {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
