"""
Measure one simulator on this machine with the same Panda pick-a-cube task: start-up time, environment steps per
second (without or with a camera image) and a short video, so MuJoCo and ManiSkill can be compared machine by machine.

Usage (from sim2real_worldmodel/; see terminal_commands.md for the macOS Vulkan variables ManiSkill needs):
    python scripts/bench_sim.py --sim robosuite --out <DIR> [--camera]
    python scripts/bench_sim.py --sim maniskill --out <DIR> [--camera] [--num-envs 256 --gpu]   # GPU: Linux + CUDA

Tasks: robosuite Lift (MuJoCo physics) and ManiSkill3 PickCube-v1 (SAPIEN / PhysX physics), both a Panda arm and one
cube, both controlled at 20 Hz with random joint-position deltas (the one control mode both have on every platform:
ManiSkill's end-effector control needs Pinocchio, missing from its macOS build). --camera adds a 128 x 128 RGB camera
image to every observation. One measurement per run (a process holds one ManiSkill GPU simulation). Appends one JSON
line to <DIR>/bench_sim.jsonl; without --camera also writes the video <DIR>/<sim>_<machine>.gif (3 s, 256 x 256).

Author: Duc Anh Hoang
"""

import argparse
import json
import os
import platform
import time

import imageio.v2 as imageio
import numpy as np
import torch

CAMERA = 128  # side of the camera image measured, in pixels
VIDEO = 256  # side of the saved video frames
CONTROL_HZ = 20


def machine() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0).replace(" ", "_")
    return f"{platform.system()}_{platform.machine()}".lower()


class Robosuite:
    def __init__(self, camera: bool):
        import robosuite
        from robosuite.controllers import load_composite_controller_config, load_part_controller_config

        config = load_composite_controller_config(controller="BASIC", robot="Panda")
        arm = load_part_controller_config(default_controller="JOINT_POSITION")
        arm["gripper"] = config["body_parts"]["right"]["gripper"]
        config["body_parts"] = {"right": arm}
        self.env = robosuite.make(
            "Lift", robots="Panda", controller_configs=config, control_freq=CONTROL_HZ, horizon=10**6,
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=camera, camera_names="agentview",
            camera_heights=CAMERA, camera_widths=CAMERA,
        )  # fmt: skip
        self.low, self.high = self.env.action_spec
        self.version = {"robosuite": robosuite.__version__, "mujoco": __import__("mujoco").__version__}

    def reset(self):
        self.env.reset()

    def step(self):
        self.env.step(np.random.uniform(self.low, self.high))

    def frame(self):
        return self.env.sim.render(camera_name="frontview", width=VIDEO, height=VIDEO)[::-1]


class ManiSkill:
    def __init__(self, camera: bool, num_envs: int = 1, gpu: bool = False):
        import gymnasium as gym
        import mani_skill
        import mani_skill.envs  # noqa: F401  (registers the tasks)

        self.env = gym.make(
            "PickCube-v1", obs_mode="rgb" if camera else "state", control_mode="pd_joint_delta_pos",
            render_mode="rgb_array", num_envs=num_envs, sim_backend="physx_cuda" if gpu else "physx_cpu",
            sensor_configs={"width": CAMERA, "height": CAMERA},
            human_render_camera_configs={"width": VIDEO, "height": VIDEO},
        )  # fmt: skip
        self.version = {"mani_skill": mani_skill.__version__, "sapien": __import__("sapien").__version__}

    def reset(self):
        self.env.reset(seed=0)

    def step(self):
        self.env.step(self.env.action_space.sample())

    def frame(self):
        return self.env.render()[0].cpu().numpy()


def steps_per_second(env, steps: int) -> float:
    env.reset()
    for _ in range(10):  # warm-up: first steps compile kernels and fill caches
        env.step()
    start = time.perf_counter()
    for _ in range(steps):
        env.step()
    return steps / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", choices=("robosuite", "maniskill"), required=True)
    parser.add_argument("--out", required=True, help="folder for bench_sim.jsonl and the video")
    parser.add_argument("--steps", type=int, default=500, help="timed control steps per measurement")
    parser.add_argument("--num-envs", type=int, default=1, help="parallel environments (ManiSkill only)")
    parser.add_argument("--gpu", action="store_true", help="ManiSkill GPU simulation (physx_cuda)")
    parser.add_argument("--camera", action="store_true", help="add a camera image to every observation")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(0)

    start = time.perf_counter()
    env = Robosuite(args.camera) if args.sim == "robosuite" else ManiSkill(args.camera, args.num_envs, args.gpu)
    env.reset()
    startup = time.perf_counter() - start  # imports, scene building and the first reset
    sps = steps_per_second(env, args.steps)
    if not args.camera:
        frames = [env.frame()]
        for _ in range(3 * CONTROL_HZ):
            env.step()
            frames.append(env.frame())
        name = f"{args.sim}_{machine()}" + (f"_gpu{args.num_envs}" if args.gpu else "")
        imageio.mimsave(os.path.join(args.out, f"{name}.gif"), frames, duration=1000 / CONTROL_HZ, loop=0)
    row = {
        "sim": args.sim, "machine": machine(), "platform": platform.platform(), "gpu_sim": args.gpu,
        "num_envs": args.num_envs, "camera": args.camera, **env.version, "startup_s": round(startup, 2),
        "steps_per_s": round(sps * args.num_envs),  # control steps x parallel environments
    }  # fmt: skip
    with open(os.path.join(args.out, "bench_sim.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
