# Region-Text Contrastive Grounding for Language-driven Grasp Detection

Given an RGB image and a text prompt such as "Lift the hairbrush by its handle.", predict one 2D grasp rectangle
(x, y, w, h, θ) for a two-finger gripper. Dataset: [Grasp-Anything++](https://airvlab.github.io/grasp-anything/)
(Vuong et al., CVPR 2024).

This folder contains:

1. a reproduction of the **GR-ConvNet + CLIP** baseline of Grasp-Anything++ (frozen CLIP text encoder, one ALBEF-style
   cross-attention layer);
2. a **failure analysis** of that baseline: an automatic sort of every failure plus two vision-language judges on 1,000
   failures, which shows that most failures grasp another object than the prompted one;
3. the proposed **region-text contrastive grounding loss**: during training, the encoder features of the prompted object
   (pooled over its mask) must match the CLIP embedding of the prompt better than the other objects of the same scene
   and the background. The loss is used in training only; the network at inference is the baseline's.

## Results

Official split of the LGD code (`split/grasp-anything++` of [Fsoft-AIC/LGD](https://github.com/Fsoft-AIC/LGD)):
10,130 training, 515 Seen and 212 Unseen test objects with released labels (28.6k / 1,376 / 601 prompts). Settings of
the authors' scripts: Adam, learning rate 1e-3, batch 8, 50 epochs of 1,000 batches, model chosen on the Seen test;
success = IoU > 25% and angle error < 30° with a ground-truth grasp (predicted opening scaled by 150, as their
`inference/post_process.py`); H = harmonic mean of Seen and Unseen. Mean ± std over seeds 123, 124, 125.

| Model | Text | Seen | Unseen | H |
|---|---|---|---|---|
| GR-ConvNet + CLIP, official code re-run ‡ | object name | (0.33 *) | 0.18 | - |
| LGD, released model, official code ‡ | object name | - | 0.29 ± 0.03 (168 random prompts) | - |
| GR-ConvNet + CLIP, ours (ALBEF fusion) | instruction | 0.331 ± 0.019 | 0.392 ± 0.017 | 0.358 |
| **+ region-text contrastive (λ = 0.01)** | instruction | **0.365 ± 0.006** | **0.403 ± 0.032** | **0.383** |

‡ Run with the authors' code and README commands, unchanged. * Their training script and Seen evaluation both read
`split/grasp-anything++/test/seen.obj`, so only Unseen is comparable. Published numbers: GR-ConvNet + CLIP 0.37 / 0.18,
LGD 0.48 / 0.42.

- The H gain (+2.5) is positive in every seed. Paired per prompt over the 3 seeds: Seen +3.5 points (sign test
  p = 1e-6); official Unseen +1.1 (p = 0.32, only 601 prompts); all 13,464 labelled Unseen prompts of the
  Grasp-Anything list +0.9 (p = 5e-5).
- Weight sensitivity: λ = 0.03 gives H 0.375 ± 0.054; λ = 0.1 collapses in one seed (the contrastive term stops the
  quality map from forming with batch 8).
- Grounding gates (mean of 3 seeds, Unseen): the grasp restricted to the prompted object's ground-truth mask (oracle)
  reaches 0.518 (baseline) and 0.565 (ours); restricted to a Grounding DINO box, 0.443 and 0.464.

Protocol details of the released LGD code, checked at commit 0223d43: its evaluation feeds the object name, not the
Grasp-Anything++ instruction; its GR-ConvNet + CLIP detaches the image features before adding the text.

Checkpoints, per-prompt predictions and logs of these runs are kept in a private Hugging Face dataset (available on
request); the runs of the earlier version below are in the Kaggle Dataset
[hoangducanh1865/fpt-entrance-test-runs](https://www.kaggle.com/datasets/hoangducanh1865/fpt-entrance-test-runs).

An earlier version of this work (our own split of the Grasp-Anything lists, 2 x T4) analysed the baseline's failures
with two vision-language judges: about half of them grasp another object than the prompted one, which motivated the
loss (`scripts/analyze_errors.py`, `scripts/judge_failures.py`).

## Layout

The code follows [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) and runs unchanged on a
local machine, Google Colab and Kaggle (the platform is detected; per-platform paths are in `configs/paths/`).

| Path | Content |
|---|---|
| `configs/` | [Hydra](https://hydra.cc) configs: `train.yaml`, `eval.yaml`; `experiment/` one file per run (only what differs); `paths/` one file per platform |
| `src/train.py`, `src/eval.py` | training and evaluation entry points |
| `src/data/grasp_anything.py` | dataset and Lightning DataModule (packed HDF5, cached CLIP text features, region masks) |
| `src/models/grconvnet.py` | GR-ConvNet with the ALBEF-style text fusion |
| `src/models/region_text_contrast.py` | **the proposed loss** |
| `src/models/grasp_module.py` | LightningModule: grasp loss + weighted contrastive loss |
| `src/utils/grasp.py` | grasp maps, decoding, success metric |
| `scripts/prepare_data.py`, `resplit_lgd.py`, `cache_text_emb.py` | pack Grasp-Anything++ into HDF5, re-split into the official LGD split, cache CLIP text features |
| `scripts/detect_boxes.py` | Grounding DINO boxes for the detector gate |
| `scripts/analyze_errors.py`, `judge_failures.py` | failure analysis and VLM judging |
| `scripts/lgd_extract.py`, `lgd_official.sh`, `lgd_shard.py` | the official LGD code re-run (data layout, commands, sharded evaluation) |
| `scripts/a100_lgd.sh`, `kaggle_pair.sh` | the jobs of the 3-seed runs (Colab A100, Kaggle 2 x T4) |
| `tests/` | unit tests (`pytest`) |
| `paper/` | LaTeX source of the paper (CVPR 2026 template) |
| `../scripts/`, `../terminal_commands.md` | remote runs on Kaggle and Colab (shared by every task of the repo); every command, per platform |

## Usage

Every command runs from this folder (`cd lang_grasp`).

```bash
pip install -r requirements.txt     # plus torch and torchvision for your platform
pytest -q

# 1. Data: download airvlab/Grasp-Anything and airvlab/Grasp-Anything-pp from Hugging Face into <DATA>/datasets/raw
python scripts/prepare_data.py --raw <DATA>/datasets/raw --out <DATA>/datasets/processed/Grasp-Anything-pp
python scripts/resplit_lgd.py --src <DATA>/datasets/processed/Grasp-Anything-pp --out <LGD>   # official split
python scripts/cache_text_emb.py --processed <LGD>

# 2. Train (baseline, then ours) and evaluate the best checkpoint
python src/train.py experiment=grconvnet_clip seed=123 paths.data_dir=<LGD>
python src/train.py experiment=grconvnet_clip_contrast model.contrast_weight=0.01 seed=123 paths.data_dir=<LGD>
python src/eval.py ckpt_path=<RUN>/checkpoints/<BEST>.ckpt paths.data_dir=<LGD> "splits=[test_seen,test_unseen,test_unseen_all]"

# 3. Grounding gates
python scripts/detect_boxes.py --data <LGD> --out <BOXES>
python src/eval.py ckpt_path=<CKPT> paths.data_dir=<LGD> gate=oracle
python src/eval.py ckpt_path=<CKPT> paths.data_dir=<LGD> gate=detector boxes=<BOXES>

# 4. Failure analysis of an evaluation, then the VLM judges (any OpenAI-compatible endpoint)
python scripts/analyze_errors.py --eval <EVAL_RUN> --data <LGD> --out <OUT> \
    --export "background=500,wrong_object=250,wrong_geometry=250"
python scripts/judge_failures.py --dir <OUT>/export --model <MODEL> --base-url <URL>
```

The official LGD code is re-run with `scripts/lgd_official.sh` (see its header for the setup: their repository at
commit 0223d43, the released checkpoint from their README, the official Hugging Face data, Python 3.9 and torch 1.12).

## Reused code

- GR-ConvNet network (`src/models/grconvnet.py`) and the grasp maps, decoding and metric (`src/utils/grasp.py`) are
  adapted from the Grasp-Anything authors' code, [Fsoft-AIC/Grasp-Anything](https://github.com/Fsoft-AIC/Grasp-Anything)
  (MIT licence), itself based on GR-ConvNet (Kumra et al., IROS 2020). Each file header says what was taken and what
  changed.
- The CLIP text encoder is `openai/clip-vit-base-patch32` from Hugging Face Transformers.
- The official LGD code ([Fsoft-AIC/LGD](https://github.com/Fsoft-AIC/LGD)) is only run, never copied into this
  repository; `scripts/lgd_shard.py` wraps its evaluation script without changing it.
- Grounding DINO (`IDEA-Research/grounding-dino-base`) from Hugging Face Transformers, for the detector gate only.
- Everything else, including the proposed loss (`src/models/region_text_contrast.py`), the data packing and the
  failure analysis, is new.

## Author

Duc Anh Hoang
