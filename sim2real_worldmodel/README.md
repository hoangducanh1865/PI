# Human-like synthetic robot data

Work in progress (started 2026-10-27). Question: does synthetic robot data, made in a simulator or by a world model,
move like human demonstrations, and does that matter for the policy trained on it? This folder will hold trajectory
features (straightness, speed profile, smoothness), distribution distances between human and generated demo sets,
and the experiments that use them.

| Path | Content |
|---|---|
| `scripts/extract_mimicgen.py` | MimicGen human source demos and generated demos reduced to end-effector, gripper and action trajectories (no images) |
| `scripts/bench_sim.py` | the same Panda pick-a-cube task in robosuite (MuJoCo) and ManiSkill3: start-up time, steps per second with and without a camera, a short video |

## Results so far

Simulator benchmark (`scripts/bench_sim.py`, 1 environment, random joint-position deltas at 20 Hz, environment steps
per second):

| Machine | robosuite 1.5.2 Lift (MuJoCo 3.3.7) | ManiSkill 3.0.1 PickCube (SAPIEN 3.0.3) |
|---|---|---|
| MacBook Air M1, state only / + 128 x 128 camera | 379 / 96 | **1,096** / **158** |
| Colab CPU (2 vCPU, software Vulkan), state only / + camera | not run (OSMesa segfault) | 240 / 17 |
| Kaggle (4 vCPU, Tesla T4), CPU simulation, state only / + camera | 106 / 102 | 295 / 214 |
| Kaggle Tesla T4, GPU simulation, 256 environments, state only / + camera | not available | 6,901 / 3,500 |
| Kaggle Tesla T4, GPU simulation, 1,024 environments, state only | not available | **17,830** |

Setup notes measured on the way: robosuite 1.5.2 breaks on MuJoCo 3.17; ManiSkill on macOS always renders, so it
needs Vulkan (MoltenVK) even without a camera; mani_skill 3.0.1 pins mplib 0.1.1, which has no Python 3.13 wheel.

Every command runs from this folder (`cd sim2real_worldmodel`); per platform: `../terminal_commands.md`.

## Author

Duc Anh Hoang
