# Human-like synthetic robot data

Work in progress (started 2026-10-27). Question: does synthetic robot data, made in a simulator or by a world model,
move like human demonstrations, and does that matter for the policy trained on it? This folder will hold trajectory
features (straightness, speed profile, smoothness), distribution distances between human and generated demo sets,
and the experiments that use them.

| Path | Content |
|---|---|
| `scripts/extract_mimicgen.py` | MimicGen human source demos and generated demos reduced to end-effector, gripper and action trajectories (no images) |
| `scripts/bench_sim.py` | the same Panda pick-a-cube task in robosuite (MuJoCo) and ManiSkill3: start-up time, steps per second with and without a camera, a short video |

Every command runs from this folder (`cd sim2real_worldmodel`); per platform: `../terminal_commands.md`.

## Author

Duc Anh Hoang
