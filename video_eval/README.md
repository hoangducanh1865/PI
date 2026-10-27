# Evaluating generated videos for robot trajectories

Work in progress (2026-10-27 to 2026-11-02). Reproduces IRASim (Zhu et al., 2024, [bytedance/IRASim](https://github.com/bytedance/IRASim),
Apache-2.0), an action-conditioned video generator for real robots, with its official Bridge checkpoints, and compares
it with the baselines released alongside it (LVDM, VDM) on visual quality, temporal consistency, physical plausibility,
task completion and robot motion.

| Path | Content |
|---|---|
| `scripts/fetch_irasim.py` | fetch only the needed checkpoints and samples from the IRASim release (tens of GB per archive) |

Every command runs from this folder (`cd video_eval`); per platform: `../terminal_commands.md`.

## Author

Duc Anh Hoang
