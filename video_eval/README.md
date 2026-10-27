# Evaluating generated videos for robot trajectories

Work in progress (2026-10-27 to 2026-11-02). Reproduces IRASim (Zhu et al., 2024, [bytedance/IRASim](https://github.com/bytedance/IRASim),
Apache-2.0), an action-conditioned video generator for real robots, with its official Bridge checkpoints, and compares
it with the baselines released alongside it (LVDM, VDM) on visual quality, temporal consistency, physical plausibility,
task completion and robot motion.

| Path | Content |
|---|---|
| `scripts/fetch_irasim.py` | fetch only the needed checkpoints and samples from the IRASim release (tens of GB per archive) |
| `scripts/setup_irasim.sh` | official IRASim code at a pinned commit, its pinned packages apart from the platform's, the SDXL VAE |
| `scripts/select_episodes.py` | the Bridge test episodes generated: the first two per instruction verb |
| `scripts/make_references.py` | the same clips as real video (ceiling) and as the still first frame (floor), with IRASim's preprocessing, plus the robot state |
| `scripts/eval_metrics.py` | PSNR, SSIM, LPIPS of generated clips against the real clips (frames 1 to 15) |
| `scripts/kaggle_batch.sh` | the whole chain on a fresh Kaggle machine (fetch, setup, choose, generate), one job queue per GPU |
| `scripts/run_irasim.py` | generate the chosen clips with one released model (Frame-Ada, Video-Ada, LVDM, VDM) and seed |

Every command runs from this folder (`cd video_eval`); per platform: `../terminal_commands.md`.

## Author

Duc Anh Hoang
