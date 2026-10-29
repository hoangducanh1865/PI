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
| LVDM | 2 (3407, 3408) | 22.65 (22.25 to 23.06) | 0.797 (0.792 to 0.801) | 0.099 (0.094 to 0.103) |
| VDM | 1 (3407) | 17.85 | 0.713 | 0.203 |
| still (floor) | - | 20.35 | **0.816** | 0.134 |

Per-clip standard deviations are about 2.5 dB (PSNR), 0.05 (SSIM) and 0.035 (LPIPS), larger than the gaps between the
three models, so their order needs the paired, per-clip comparison (to come). SSIM does not separate the models from a
video in which nothing moves, and in 2 of 24 clips the still frame has a lower LPIPS than Frame-Ada: pixel metrics
alone cannot say whether the robot follows the given actions. VDM scores below the still floor on all three (blurred, broken arm and a colour shift; whether the shift comes
from the model or from our run is being checked against the paper). More seeds and the motion metrics (end-effector
track against the real clip) are being added.

### Larger set and second dataset (the same scripts, Kaggle batch kernels)

Bridge, 105 clips (6 episodes per verb over 12 verbs; the 24 clips above are included). ee_* on the 96 clips with an
end-effector point (not yet VLM-checked). Paired Wilcoxon tests over the same clips:
Frame-Ada > Video-Ada on PSNR (p = 2e-5, 69% of clips) and ee_ndtw (p = 0.02); Frame-Ada > LVDM on PSNR, LPIPS and
ee_ndtw (p < 1e-8); the two Frame-Ada seeds do not differ (p = 0.15 to 0.4).

| Model | PSNR | SSIM | LPIPS | ee_ndtw | ee error (px) | moving_ndtw |
|---|---|---|---|---|---|---|
| IRASim-Frame-Ada, seed 3407 | **23.95** | **0.809** | **0.082** | <u>0.699</u> | 10.6 | **0.780** |
| IRASim-Frame-Ada, seed 3408 | <u>23.78</u> | <u>0.807</u> | <u>0.084</u> | **0.719** | **9.2** | <u>0.777</u> |
| IRASim-Video-Ada | 23.52 | 0.804 | 0.087 | 0.682 | <u>10.5</u> | 0.749 |
| LVDM | 22.62 | 0.788 | 0.103 | 0.593 | 14.9 | 0.669 |
| still (floor) | 19.80 | 0.805 | 0.141 | 0.156 | 58.7 | 0.324 |

Language-Table, 57 clips of 48 episodes (288 x 512; 2-D pushing; the end-effector point is fitted to the summed 2-D
actions, R^2 0.87). Frame-Ada > LVDM on PSNR (p = 0.015) and ee_ndtw (p = 0.017). Video-Ada (regenerated alone after
its first run ran out of memory; checkpoint loaded without errors) is far behind on this dataset: below LVDM on every
score (p < 0.001) and close to the still floor on motion; whether the paper reports the same gap is still to be checked.

| Model | PSNR | SSIM | LPIPS | ee_ndtw | ee error (px) | moving_ndtw |
|---|---|---|---|---|---|---|
| IRASim-Frame-Ada, seed 3407 | <u>27.03</u> | 0.861 | **0.059** | **0.715** | **7.9** | **0.716** |
| IRASim-Frame-Ada, seed 3408 | **27.27** | 0.862 | **0.059** | <u>0.688</u> | <u>10.6</u> | <u>0.690</u> |
| IRASim-Video-Ada | 23.04 | 0.833 | 0.104 | 0.304 | 37.2 | 0.355 |
| LVDM | 26.44 | <u>0.863</u> | 0.066 | 0.646 | 10.9 | 0.642 |
| still (floor) | 22.35 | **0.879** | 0.103 | 0.131 | 80.7 | 0.228 |

On Language-Table, SSIM ranks the still frame above every model (wide static table); the motion scores do not.

### Cosmos-Predict2.5 and the 3-D action score (A100, `scripts/run_cosmos.py`, `scripts/cosmos3_idm.py`)

3-D score: NVIDIA Cosmos3-Nano reads the end-effector motion back from each video (inverse dynamics, no training of
ours), decoded exactly as NVIDIA's cosmos-framework encodes Bridge actions; the recovered path is compared with the
commanded one (cm, no real video needed). Its own error on the real videos is 1.50 cm (clips move about 25 cm).

Bridge, 105 clips (Cosmos base: the 24 core clips only):

| Model | PSNR | SSIM | LPIPS | ee_ndtw | ee error (px) | DINOv2 adjacent | 3-D error (cm) | ndtw_3d |
|---|---|---|---|---|---|---|---|---|
| real (ceiling) | - | - | - | 1 | 0 | 0.952 | 1.50 | 0.528 |
| Cosmos-Predict2.5 action | **25.74** | **0.859** | **0.063** | **0.754** | **7.4** | 0.929 | **1.64** | **0.498** |
| IRASim-Frame-Ada, seed 3407 | <u>23.95</u> | <u>0.809</u> | <u>0.082</u> | 0.699 | 10.6 | 0.934 | <u>1.81</u> | <u>0.476</u> |
| IRASim-Frame-Ada, seed 3408 | 23.78 | 0.807 | 0.084 | <u>0.719</u> | <u>9.2</u> | 0.935 | 1.84 | <u>0.476</u> |
| IRASim-Video-Ada | 23.52 | 0.804 | 0.087 | 0.682 | 10.5 | 0.933 | 1.95 | 0.450 |
| LVDM | 22.62 | 0.788 | 0.103 | 0.593 | 14.9 | 0.925 | 2.54 | 0.377 |
| Cosmos-Predict2.5 base (no actions, 24 clips) | 13.55 | 0.611 | 0.290 | 0.131 | 64.0 | 0.916 | 10.30 | 0.041 |
| still (floor) | 19.80 | 0.805 | 0.141 | 0.156 | 58.7 | 1.000 | 9.06 | 0.045 |

Paired Wilcoxon: Cosmos action > Frame-Ada on PSNR (p = 4e-16), ee_ndtw (p = 0.008) and the 3-D error (p = 0.02);
Frame-Ada > Video-Ada > LVDM on the 3-D error (p <= 0.01 each). Without actions the same backbone is no better than the
still video on the 3-D error (p = 0.32), although its frames are smooth.

3-D score on the 24 core clips (Frame-Ada seed 3407 unless stated): real 1.68, original actions 1.93, noise 0.5 cm
2.67, noise 1.5 cm 5.62, frozen actions 9.73, still video 9.52, mirrored 15.32; six Frame-Ada seeds 1.86 to 2.12;
Video-Ada 2.09 / 2.13, LVDM 2.15 / 2.53, VDM 10.68. Best of six seeds chosen by the 3-D score: PSNR 24.02 vs 23.93
for the average seed (p = 0.64), ee_ndtw 0.717 vs 0.709 (p = 0.37): no measurable gain, the seeds differ little.

### Motion (`scripts/eval_motion.py --tracker cotracker`; points tracked from frame 0 in the generated and real clip)

End-effector track (the tracked point best explained by the recorded end-effector position), on the 13 clips where two
VLM judges (`scripts/check_ee_points.py`, gemini-3.8-flash and gpt-6.1-sol) agreed, in two separate runs, that the point
lies on the gripper:

| Model | ee_ndtw | mean error (px) | final error (px) | path_ratio (real = 1) | moving_ndtw |
|---|---|---|---|---|---|
| IRASim-Frame-Ada (6 seeds) | **0.767** | **6.3** | **10.7** | 1.00 | **0.792** |
| IRASim-Video-Ada (2 seeds) | <u>0.725</u> | <u>7.6</u> | <u>12.0</u> | 0.98 | <u>0.758</u> |
| LVDM (2 seeds) | 0.609 | 12.6 | 22.4 | 0.98 | 0.693 |
| VDM (1 seed) | 0.169 | 55.1 | 96.0 | 0.37 | 0.367 |
| still (floor) | 0.135 | 58.5 | 101.5 | 0.00 | 0.342 |

NDTW = exp(-DTW / (16 * 20 px)); moving_ndtw averages it over every point that moves in the real clip. The ranking is
the same on all 22 clips with an end-effector point (Frame-Ada 0.709, Video-Ada 0.656, LVDM 0.568, VDM 0.173, still
0.152). Unlike SSIM, the motion scores put every model far from the still floor except VDM, whose robot barely moves.
Two lessons on the measurement itself: (1) with a weaker tracker (Lucas-Kanade) the blurred gripper is lost, the chosen
point lands on the cable or the held object, and the models seem to move a third less than the real robot (path ratio
0.66); CoTracker3 removes that artefact (1.00); (2) the VLM judges are not stable at temperature 0: run to run, 2 of 15
verified clips changed, so only clips verified in both runs are kept.

### Perturbation test (does a metric notice wrong actions?)

IRASim-Frame-Ada, seed 3407, the same 24 clips, generated from corrupted actions and scored against the real clips of
the original actions (`scripts/make_perturbed_annotations.py`): random-walk noise of 0.5 and 1.5 cm per frame on the
end-effector position, mirrored motion, frozen (all actions zero). ee_* on the 13 verified clips.

| Actions given | PSNR | SSIM | LPIPS | ee_ndtw | ee error (px) | path_ratio |
|---|---|---|---|---|---|---|
| original | **23.97** | <u>0.814</u> | **0.081** | **0.759** | **6.2** | 0.98 |
| noise 0.5 cm | <u>21.87</u> | 0.788 | <u>0.103</u> | <u>0.550</u> | <u>15.2</u> | 1.09 |
| noise 1.5 cm | 19.45 | 0.752 | 0.149 | 0.270 | 39.0 | 1.50 |
| frozen | 19.31 | 0.753 | 0.158 | 0.133 | 59.1 | 0.21 |
| mirrored | 18.59 | 0.737 | 0.178 | 0.036 | 99.3 | 0.92 |
| still frame (floor) | 20.35 | **0.816** | 0.134 | 0.135 | 58.5 | 0.00 |

ee_ndtw falls with the corruption and puts frozen actions exactly at the still floor (0.133 vs 0.135) and mirrored
motion below it; PSNR and SSIM rank the still frame above the corrupted generations (SSIM even above the original).
Regenerating seed 3407 on another machine gave identical videos (PSNR difference 0.0): generation is deterministic.

### Temporal consistency (`scripts/eval_temporal.py`, 24 clips)

| Model | DINOv2 adjacent | DINOv2 to frame 0 | warp error (0-255) |
|---|---|---|---|
| real | 0.953 | 0.854 | 3.35 |
| IRASim-Frame-Ada | 0.933 | 0.842 | **4.42** |
| IRASim-Video-Ada | 0.938 | 0.842 | <u>4.52</u> |
| LVDM | 0.924 | 0.829 | 4.73 |
| VDM | 0.931 | 0.789 | 5.00 |
| still | 1.000 | 1.000 | 0.01 |

The still clip scores perfectly: these measure stability, not correct motion.

### Physical plausibility and outcome, two VLM judges (`scripts/eval_vlm.py`, 24 clips; gemini-3.8-flash / gpt-6.1-sol)

| Model | judged plausible | same outcome as the real clip | most frequent violation |
|---|---|---|---|
| real (control) | 0.79 / 0.96 | 1.00 / 1.00 | |
| IRASim-Frame-Ada | 0.33 / 0.83 | 0.50 / 0.75 | object changes or vanishes (0.46 / 0.17) |
| IRASim-Video-Ada | 0.42 / 0.88 | **0.62 / 0.75** | object changes or vanishes (0.42 / 0.08) |
| LVDM | 0.42 / 0.75 | 0.29 / 0.67 | object changes or vanishes (0.42 / 0.25) |
| VDM | 0.04 / 0.25 | 0.04 / 0.08 | robot deforms or vanishes (0.96 / 0.71) |
| still (control) | 1.00 / 1.00 | 0.00 / 0.04 | |

The judges differ in strictness (gemini flags 21% of real clips, gpt 4%; they agree on 46 to 58% of IRASim clips) but
rank the models alike; the still clip is "plausible" every time, and only the outcome question catches it.

## Layout

| Path | Content |
|---|---|
| `scripts/fetch_irasim.py` | fetch only the needed checkpoints and samples from the IRASim release (tens of GB per archive) |
| `scripts/setup_irasim.sh` | official IRASim code at a pinned commit, its pinned packages apart from the platform's, the SDXL VAE |
| `scripts/select_episodes.py` | the Bridge test episodes generated: the first two per instruction verb |
| `scripts/make_references.py` | the same clips as real video and as the still first frame, with IRASim's preprocessing, plus the robot state |
| `scripts/setup_cosmos.sh`, `scripts/run_cosmos.py` | NVIDIA Cosmos-Predict2.5-2B (action-conditioned on Bridge, and image + text) through the official repo, stored like the IRASim clips |
| `scripts/idm.py` | our inverse dynamics models (from scratch on SDXL latents, or fine-tuned DINOv2): recover a clip's actions, integrated into a 3-D end-effector path and compared with the commanded one |
| `scripts/setup_cosmos3.sh`, `scripts/cosmos3_idm.py` | the same 3-D scores with NVIDIA Cosmos3-Nano's own inverse dynamics (served by vLLM-Omni; linear calibration with 2-fold cross-fitting on real clips) |
| `scripts/a100_job.sh` | the GPU jobs a T4 cannot do (Cosmos, needs Ampere) and the 3-D scores (Cosmos3-Nano inverse dynamics; our IDMs only as the fallback), on one A100 machine |
| `scripts/run_irasim.py` | generate the chosen clips with one released model and seed |
| `scripts/make_perturbed_annotations.py` | corrupted action inputs (position noise, mirrored motion, frozen) for the perturbation test |
| `scripts/kaggle_batch.sh` | the whole chain on a fresh Kaggle machine (fetch, setup, choose, references, generate, score), one job queue per GPU, optional corrupted actions |
| `scripts/eval_metrics.py` | PSNR, SSIM, LPIPS against the real clips |
| `scripts/eval_temporal.py` | temporal consistency: DINOv2 frame similarity, optical-flow warp error |
| `scripts/eval_vlm.py` | physical plausibility and task outcome by VLM judges (`scripts/vlm.py`: shared client) |
| `scripts/eval_motion.py` | motion against the real clips: tracked points, end-effector track, NDTW, path ratio |
| `scripts/check_ee_points.py` | VLM judges check that the chosen end-effector point lies on the gripper |

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
