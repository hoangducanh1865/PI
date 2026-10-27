"""
Cache the CLIP text features of every prompt once; the text encoder stays frozen during training.

Usage (Colab or Kaggle; a CPU runtime is enough, a GPU is faster):
    python scripts/cache_text_emb.py --processed <DIR> [--model openai/clip-vit-base-patch32] [--batch 256]
    e.g. python scripts/cache_text_emb.py \
             --processed /content/drive/MyDrive/it/fpt/data/datasets/processed/Grasp-Anything-pp

For every <split>.h5 in <DIR> it writes <split>_text_<tag>.npz (<tag> = the model name after the "/"), in the
order of <split>.h5:
    emb         float32 [n, 512]     CLIP's projected (pooled) text embedding, not normalised
    tokens      float16 [u, L, 512]  last hidden states of the u unique prompts (the token sequence)
    token_mask  bool [u, L]          True for real tokens, False for padding
    text_index  int32 [n]            sample i uses tokens[text_index[i]]
    keys        str [n]

Author: Duc Anh Hoang
"""

import argparse
import glob
import os
import time

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed", required=True, help="folder with the <split>.h5 files of prepare_data.py")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = CLIPTextModelWithProjection.from_pretrained(args.model).to(device).eval()
    tag = args.model.split("/")[-1]
    for path in sorted(glob.glob(os.path.join(args.processed, "*.h5"))):
        start = time.time()
        with h5py.File(path) as f:
            keys = [k.decode() for k in f["keys"][:]]
            prompts = [p.decode() for p in f["prompts"][:]]
        unique = sorted(set(prompts))  # many prompts repeat; encode each text once
        tokens = tokenizer(unique, padding=True, truncation=True, return_tensors="pt")
        emb, hidden = [], []
        with torch.no_grad():
            for i in range(0, len(unique), args.batch):
                out = model(**{k: v[i : i + args.batch].to(device) for k, v in tokens.items()})
                emb.append(out.text_embeds.float().cpu().numpy())
                hidden.append(out.last_hidden_state.half().cpu().numpy())
        position = {p: i for i, p in enumerate(unique)}
        text_index = np.array([position[p] for p in prompts], dtype=np.int32)
        out_path = path[: -len(".h5")] + f"_text_{tag}.npz"
        np.savez(
            out_path,
            emb=np.concatenate(emb)[text_index],
            tokens=np.concatenate(hidden),
            token_mask=tokens["attention_mask"].numpy().astype(bool),
            text_index=text_index,
            keys=np.array(keys),
        )
        print(
            f"{os.path.basename(out_path)}: {len(prompts)} prompts ({len(unique)} unique, "
            f"{tokens['input_ids'].shape[1]} tokens max), {time.time() - start:.0f} s on {device}",
            flush=True,
        )


if __name__ == "__main__":
    main()
