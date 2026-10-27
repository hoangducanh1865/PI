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

Both models trained with the same settings on 2 x T4 (data parallel, effective batch 16, 50 epochs), evaluated on our
split of Grasp-Anything++ (3,000 Seen and 13,464 Unseen test prompts). Success = IoU > 25% and angle difference
< 30° with a ground-truth grasp; H = harmonic mean of Seen and Unseen.

| Model | Seen | Unseen | H |
|---|---|---|---|
| GR-ConvNet + CLIP, reproduced | 0.407 | 0.260 | 0.317 |
| + region-text contrastive (ours) | **0.427** | **0.335** | **0.375** |

Outcome of each test prompt (failure types judged by two VLMs on 1,000 failures per model, 87% agreement):

| Share of prompts | Baseline Seen | Baseline Unseen | Ours Seen | Ours Unseen |
|---|---|---|---|---|
| Success | 40.7% | 26.0% | 42.7% | 33.5% |
| Wrong object | 27.4% | 35.5% | 26.8% | 33.7% |
| Right object, wrong geometry | 20.8% | 24.0% | 22.2% | 23.9% |
| No grasp predicted | 8.9% | 11.5% | 5.1% | 4.9% |

Most of the gain comes from fewer missed predictions; wrong-object grasps drop only slightly. Single seed per model,
λ = 0.1 and τ = 0.07 not tuned.

Checkpoints, per-prompt predictions and training logs of both runs: Kaggle Dataset
[hoangducanh1865/fpt-entrance-test-runs](https://www.kaggle.com/datasets/hoangducanh1865/fpt-entrance-test-runs)
(`grconvnet_clip/2026-10-26_12-29-17`, `grconvnet_clip_contrast/2026-10-26_09-53-17`, evaluations under `eval/`).

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
| `scripts/` | data packing, CLIP caching, error analysis, VLM judging |
| `tests/` | unit tests (`pytest`) |
| `paper/` | LaTeX source of the paper (CVPR 2026 template) |
| `../scripts/`, `../terminal_commands.md` | remote runs on Kaggle and Colab (shared by every task of the repo); every command, per platform |

## Usage

Every command runs from this folder (`cd lang_grasp`).

```bash
pip install -r requirements.txt     # plus torch and torchvision for your platform
pytest -q

# 1. Data: download Grasp-Anything and Grasp-Anything++ from Hugging Face into <DATA>/datasets/raw, then
python scripts/prepare_data.py --raw <DATA>/datasets/raw --out <DATA>/datasets/processed/Grasp-Anything-pp
python scripts/cache_text_emb.py --processed <DATA>/datasets/processed/Grasp-Anything-pp

# 2. Train (baseline, then ours) and evaluate the best checkpoint
python src/train.py experiment=grconvnet_clip
python src/train.py experiment=grconvnet_clip_contrast
python src/eval.py ckpt_path=<RUN>/checkpoints/<BEST>.ckpt

# 3. Failure analysis of an evaluation, then the VLM judges (any OpenAI-compatible endpoint)
python scripts/analyze_errors.py --eval <EVAL_RUN> --data <DATA>/datasets/processed/Grasp-Anything-pp --out <OUT> \
    --export "background=500,wrong_object=250,wrong_geometry=250"
python scripts/judge_failures.py --dir <OUT>/export --model <MODEL> --base-url <URL>
```

The split uses only the authors' files (`split/grasp-anything/{seen,unseen}.obj`): seen objects are cut 90% / 10% into
training and Seen test as in their training script, unseen objects form the Unseen test; one prompt is one sample.

## Reused code

- GR-ConvNet network (`src/models/grconvnet.py`) and the grasp maps, decoding and metric (`src/utils/grasp.py`) are
  adapted from the Grasp-Anything authors' code, [Fsoft-AIC/Grasp-Anything](https://github.com/Fsoft-AIC/Grasp-Anything)
  (MIT licence), itself based on GR-ConvNet (Kumra et al., IROS 2020). Each file header says what was taken and what
  changed.
- The CLIP text encoder is `openai/clip-vit-base-patch32` from Hugging Face Transformers.
- Everything else, including the proposed loss (`src/models/region_text_contrast.py`), the data packing and the
  failure analysis, is new.

## Author

Duc Anh Hoang
