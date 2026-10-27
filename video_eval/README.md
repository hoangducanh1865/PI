# Evaluating generated videos for robot trajectories

Assignment 2 of the FPT Frontier Innovators 2026 programme (Dr. Ngan Le), in progress. Question: when a video model
generates a robot executing a given action trajectory, does a realistic-looking video also show the right, physically
plausible motion? Approach: reproduce [IRASim](https://github.com/bytedance/IRASim) (Zhu et al., 2024, Apache-2.0), an
action-conditioned video generator for real robots, with its official Bridge checkpoints, run the baselines released
with it on the same clips, and score every model against two references: the real video of the same actions (ceiling)
and the first frame held still (floor).

## Models (all from the IRASim release, Bridge, 256 x 320, 16 frames, PNDM 50 steps)

| Model | Conditioning on the 15 actions (7-D end-effector deltas + gripper) | Notes |
|---|---|---|
| IRASim-Frame-Ada | per-frame: each action modulates its own frame (adaptive scale and shift) | main model; 4 seeds for best-of-N |
| IRASim-Video-Ada | video-level: the action sequence modulates the whole clip | |
| LVDM | latent video diffusion baseline (UNet) | |
| VDM | pixel-space video diffusion baseline | no VAE; one clip at a time on a T4 |
| real | the recorded video of the same actions | ceiling |
| still | first frame repeated 16 times | floor: "nothing moves" |

Clips: 24 clips of 16 frames from 16 Bridge test episodes, the first two episodes per instruction verb (put, move, take,
open, close, flip, fold, wipe). Generation uses the authors' code at commit `c72b6da` and their evaluation configs,
with only the episode subset, the seed and the batch size overridden (`scripts/run_irasim.py` lists them). The
IRASim code is cloned and run as is, outside this folder; nothing of it is copied here.

## Results so far (frames 1 to 15 against the real clip; mean over 24 clips)

| Model | Seeds | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|---|
| IRASim-Frame-Ada | 6 (3407 to 3412) | **23.94** (per seed 23.77 to 24.13) | <u>0.814</u> (0.813 to 0.817) | **0.081** (0.079 to 0.083) |
| IRASim-Video-Ada | 2 (3407, 3408) | <u>23.49</u> (23.41 to 23.57) | 0.811 (both) | <u>0.083</u> (0.081 to 0.085) |
| LVDM | 1 (3408) | 23.06 | 0.801 | 0.094 |
| VDM | 1 (3407) | 17.85 | 0.713 | 0.203 |
| still (floor) | - | 20.35 | **0.816** | 0.134 |

Per-clip standard deviations are about 2.5 dB (PSNR), 0.05 (SSIM) and 0.035 (LPIPS), larger than the gaps between the
three models, so their order needs the paired, per-clip comparison (to come). SSIM does not separate the models from a
video in which nothing moves, and in 2 of 24 clips the still frame has a lower LPIPS than Frame-Ada: pixel metrics
alone cannot say whether the robot follows the given actions. VDM scores below the still floor on all three (blurred, broken arm and a colour shift; whether the shift comes
from the model or from our run is being checked against the paper). More seeds and the motion metrics (end-effector
track against the real clip) are being added.

## Layout

| Path | Content |
|---|---|
| `scripts/fetch_irasim.py` | fetch only the needed checkpoints and samples from the IRASim release (tens of GB per archive) |
| `scripts/setup_irasim.sh` | official IRASim code at a pinned commit, its pinned packages apart from the platform's, the SDXL VAE |
| `scripts/select_episodes.py` | the Bridge test episodes generated: the first two per instruction verb |
| `scripts/make_references.py` | the same clips as real video and as the still first frame, with IRASim's preprocessing, plus the robot state |
| `scripts/run_irasim.py` | generate the chosen clips with one released model and seed |
| `scripts/kaggle_batch.sh` | the whole chain on a fresh Kaggle machine (fetch, setup, choose, generate), one job queue per GPU |
| `scripts/eval_metrics.py` | PSNR, SSIM, LPIPS against the real clips |

## Usage

From this folder, on a machine with an NVIDIA GPU (tested on a Kaggle T4; about 1.5 min per clip for IRASim-XL):

```bash
python scripts/fetch_irasim.py get bridge_checkpoints_data --out /tmp/irasim --match 'frame_ada/.*[.]pt$'
python scripts/fetch_irasim.py get bridge_train_data --out /tmp/irasim --match '/test/'     # Bridge test inputs
bash scripts/setup_irasim.sh /tmp/IRASim /tmp/irasim
python scripts/select_episodes.py --data /tmp/irasim --out <OUT>
PYTHONPATH=/tmp/IRASim/pkgs python scripts/make_references.py --data /tmp/irasim --out <OUT>
python scripts/run_irasim.py --irasim /tmp/IRASim --model frame_ada --seed 3407 --out <OUT>
pip install -r requirements.txt && python scripts/eval_metrics.py --gen <OUT> --models frame_ada/seed3407 still
```

Per platform (Kaggle interactive or batch, Colab): `../terminal_commands.md`, section video_eval.

## Author

Duc Anh Hoang
