# Demo commands

Each command saves one PNG in the `demo` folder on Drive: `/content/drive/MyDrive/it/fpt/data/lang_grasp/demo`.
Run them on Colab, in one terminal, about 10 to 25 s each.

## Setup

Once per terminal:

```bash
cd /content/drive/MyDrive/it/fpt/code/fpt_entrance_test/lang_grasp
D=/content/drive/MyDrive/it/fpt/data
export DEMO_DATA=$D/datasets/processed/Grasp-Anything-pp-lgd
export DEMO_OUT=$D/lang_grasp/demo
export DEMO_BOXES=$D/lang_grasp/a100_lgd/boxes
RUNS=$D/lang_grasp/a100_lgd/runs
BASE=$(ls $RUNS/grconvnet_clip_s125/checkpoints/epoch*.ckpt)
RTC=$(ls $RUNS/grconvnet_clip_contrast_l001_s125/checkpoints/epoch*.ckpt)
```

| Placeholder | Value |
|---|---|
| `<CKPT>` | `$BASE` (baseline) or `$RTC` (RTC-Grasp) |
| `<SPLIT>` | `test_seen` or `test_unseen` |
| `<INDEX>` | sample number: 0 to 1375 for `test_seen`, 0 to 600 for `test_unseen` |

## 1. Prompt to grasp

The grasp (yellow), the correct grasps (red) and the four output maps.

```bash
python scripts/demo.py --ckpt <CKPT> --sample <SPLIT>:<INDEX>
```

```bash
python scripts/demo.py --ckpt $RTC --sample test_unseen:498
```

Prints `RTC-Grasp: success at x=25, y=74`. Saves `test_unseen_498.png`.

## 2. Baseline against RTC-Grasp

Two checkpoints, one row each.

```bash
python scripts/demo.py --ckpt <CKPT_A> <CKPT_B> --sample <SPLIT>:<INDEX>
```

```bash
python scripts/demo.py --ckpt $BASE $RTC --sample test_unseen:498
```

The baseline fails and RTC-Grasp succeeds. Saves `test_unseen_498_compare.png`.

## 3. Same image, another prompt

Does the grasp follow the prompt? The run is not scored: the correct grasps belong to the sample's own prompt.

```bash
python scripts/demo.py --ckpt <CKPT> --sample <SPLIT>:<INDEX> --prompt "<TEXT>"
```

```bash
python scripts/demo.py --ckpt $RTC --sample test_unseen:498 --prompt "Pick up bowl by its rim."
```

The grasp stays on the banana, at `x=24, y=75`. Saves `test_unseen_498_pick_up_bowl_by_its_rim.png`.

## 4. Any image

```bash
python scripts/demo.py --ckpt <CKPT> --image <FILE> --prompt "<TEXT>"
```

```bash
python scripts/demo.py --ckpt $RTC --image $DEMO_OUT/inputs/banana_bowl.png --prompt "Pick up banana."
```

Saves `banana_bowl_pick_up_banana.png`.

## 5. Restrict the grasp to a region

`detector` uses the Grounding DINO box, `oracle` the ground-truth mask. The ruled-out area is darkened.

```bash
python scripts/demo.py --ckpt <CKPT> --sample <SPLIT>:<INDEX> --gate <detector|oracle>
```

```bash
python scripts/demo.py --ckpt $RTC --sample test_unseen:454
python scripts/demo.py --ckpt $RTC --sample test_unseen:454 --gate detector
```

Without a gate the grasp lands on the candle (failure); with the box it lands on the pumpkin (success). Saves
`test_unseen_454.png` and `test_unseen_454_detector.png`.

## 6. Regions of the contrastive loss

Each region of the image, with the probability that the prompt means it. The probabilities need `$RTC`.

```bash
python scripts/demo.py --ckpt <CKPT> --sample <SPLIT>:<INDEX> --regions
```

```bash
python scripts/demo.py --ckpt $RTC --sample test_unseen:498 --regions
```

Prints `Positive: 98%`, then the other regions. Saves `test_unseen_498_regions.png`.
