"""
Inverse dynamics model (IDM): recover the robot actions a video shows, to score generated clips in 3-D instead of in
the image. Trained on real Bridge videos; then the actions it reads from a generated clip are integrated into an
end-effector path in metres and compared with the path the model was asked to follow.

Usage (from video_eval/, on a GPU; data from scripts/fetch_irasim.py):
    python scripts/idm.py train --data <DATA_DIR> --ckpt <FILE.pt> [--backbone dinov2] [--epochs 3]
    python scripts/idm.py score --ckpt <FILE.pt> --gen <GEN_DIR> --models real still <MODEL_DIR> [...]
        [--backbone dinov2]

Actions follow IRASim (and Cosmos): between frames t-1 and t, the position change in the frame of the end effector at
t-1, R_{t-1}^T (p_t - p_{t-1}), the rotation change as Euler angles of R_{t-1}^T R_t (R = Rz Ry Rx, scipy 'xyz'), and
the gripper state at t; arm parts x 20. Two backbones, both trained with mean squared error:
- latent (from scratch): the SDXL-VAE latents of frames t-1 and t (2 x 4 x 32 x 40, the latents IRASim ships for its
  training set; at scoring time the frames are encoded with the same VAE); four strided convolutions and an MLP.
- dinov2 (fine-tuned): facebook/dinov2-small on each frame at 224 x 224 (CLS and mean patch features), the two frames'
  features and their difference into an MLP; all weights trained, backbone at a 100 x lower learning rate; trained on
  the decoded Bridge videos of the first --episodes training episodes (memory).
score: per clip, the 15 recovered actions are integrated from the real pose at frame 0 into 16 positions; compared with
the recorded positions of the real clip: mean and final error (cm), 3-D NDTW (d_th = 2 cm), path-length ratio, and
the error of the recovered actions against the given ones. The real clips give the IDM's own error (the floor of
what this score can resolve). Writes <GEN_DIR>/metrics_idm_<backbone>.csv and
<GEN_DIR>/<MODEL>/idm_<backbone>/<clip>.npz.

Author: Duc Anh Hoang
"""

import argparse
import csv
import glob
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

SCALE = np.array([20.0] * 6 + [1.0])
D_TH = 0.02  # metres
FIELDS = ("pos_err_cm", "final_err_cm", "ndtw_3d", "path_ratio_3d", "action_err")


def actions_from_states(state: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """[T-1, 7] scaled actions from [T, 6] poses (xyz, rpy) and [T] gripper states, as IRASim computes them."""
    rot = Rotation.from_euler("xyz", state[:, 3:6]).as_matrix()
    out = np.zeros((len(state) - 1, 7))
    for k in range(1, len(state)):
        out[k - 1, :3] = rot[k - 1].T @ (state[k, :3] - state[k - 1, :3])
        out[k - 1, 3:6] = Rotation.from_matrix(rot[k - 1].T @ rot[k]).as_euler("xyz")
        out[k - 1, 6] = gripper[k]
    return out * SCALE


def integrate(first: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """[T, 3] positions from the first pose (xyz, rpy) and [T-1, 7] scaled actions."""
    actions = actions / SCALE
    position, rot = first[:3].copy(), Rotation.from_euler("xyz", first[3:6]).as_matrix()
    path = [position.copy()]
    for a in actions:
        position = position + rot @ a[:3]
        rot = rot @ Rotation.from_euler("xyz", a[3:6]).as_matrix()
        path.append(position.copy())
    return np.array(path)


class IDM(nn.Module):
    def __init__(self):
        super().__init__()

        def block(cin, cout, stride):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.SiLU())

        self.net = nn.Sequential(
            block(8, 64, 1), block(64, 128, 2), block(128, 256, 2), block(256, 256, 2), nn.Flatten(),
            nn.Linear(256 * 4 * 5, 512), nn.SiLU(), nn.Linear(512, 7),
        )  # fmt: skip

    def forward(self, pair: torch.Tensor) -> torch.Tensor:  # [B, 8, 32, 40] -> [B, 7]
        return self.net(pair)


class DinoIDM(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
        self.head = nn.Sequential(nn.Linear(3 * 768, 512), nn.SiLU(), nn.Linear(512, 7))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def features(self, frames: torch.Tensor) -> torch.Tensor:  # [B, 3, 224, 224] uint8 -> [B, 768]
        hidden = self.backbone(pixel_values=(frames.float() / 255 - self.mean) / self.std).last_hidden_state
        return torch.cat([hidden[:, 0], hidden[:, 1:].mean(1)], dim=-1)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:  # [B, 6, 224, 224] uint8 -> [B, 7]
        a, b = self.features(pair[:, :3]), self.features(pair[:, 3:])
        return self.head(torch.cat([a, b, b - a], dim=-1))


def resize(video: np.ndarray) -> torch.Tensor:  # [T, H, W, 3] -> [T, 3, 224, 224] uint8
    frames = torch.from_numpy(video).permute(0, 3, 1, 2)
    return nn.functional.interpolate(frames.float(), size=(224, 224), mode="bilinear", antialias=True).round().byte()


def load_video_pairs(data: str, split: str, episodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """DINOv2 training data: [N, 6, 224, 224] uint8 frame pairs and [N, 7] actions."""
    root = os.path.join(data, "opensource_robotdata", "bridge")
    pairs, targets = [], []
    for path in sorted(glob.glob(os.path.join(root, "annotation", split, "*.json")))[:episodes]:
        with open(path) as f:
            label = json.load(f)
        video_path = os.path.join(root, label["videos"][0]["video_path"])
        if not os.path.exists(video_path):
            continue
        frames = resize(np.stack(imageio.mimread(video_path, memtest=False)))
        state, gripper = np.array(label["state"])[:, :6], np.array(label["continuous_gripper_state"])
        n = min(len(frames), len(state))
        pairs.append(torch.cat([frames[: n - 1], frames[1:n]], dim=1))
        targets.append(torch.from_numpy(actions_from_states(state[:n], gripper[:n])).float())
    return torch.cat(pairs), torch.cat(targets)


def load_pairs(data: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    root = os.path.join(data, "opensource_robotdata", "bridge")
    latents, targets = [], []
    for path in sorted(glob.glob(os.path.join(root, "annotation", split, "*.json"))):
        with open(path) as f:
            label = json.load(f)
        latent_path = os.path.join(root, label["latent_videos"][0]["latent_video_path"])
        if not os.path.exists(latent_path):
            continue
        latent = torch.load(latent_path, map_location="cpu", weights_only=True).half()  # [T, 4, 32, 40]
        state, gripper = np.array(label["state"])[:, :6], np.array(label["continuous_gripper_state"])
        n = min(len(latent), len(state))
        latents.append(torch.cat([latent[: n - 1], latent[1:n]], dim=1))
        targets.append(torch.from_numpy(actions_from_states(state[:n], gripper[:n])).float())
    return torch.cat(latents), torch.cat(targets)


def train(args) -> None:
    device = "cuda"
    if args.backbone == "latent":
        pairs, targets = load_pairs(args.data, "train")
        val_pairs, val_targets = load_pairs(args.data, "val")
        model = IDM().to(device)
        groups = [{"params": model.parameters(), "lr": 1e-3}]
    else:
        pairs, targets = load_video_pairs(args.data, "train", args.episodes)
        val_pairs, val_targets = load_video_pairs(args.data, "val", args.episodes // 10)
        model = DinoIDM().to(device)
        groups = [{"params": model.backbone.parameters(), "lr": 1e-5}, {"params": model.head.parameters(), "lr": 1e-3}]
    print(f"{args.backbone}: train pairs {len(pairs)}, val pairs {len(val_pairs)}", flush=True)
    optim = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = args.epochs * (len(pairs) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, [g["lr"] for g in groups], total_steps=steps)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(pairs))
        for i in range(len(pairs) // args.batch):
            idx = order[i * args.batch : (i + 1) * args.batch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = nn.functional.mse_loss(model(pairs[idx].to(device)).float(), targets[idx].to(device))
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(val_pairs[i : i + 256].to(device)).float().cpu()
                              for i in range(0, len(val_pairs), 256)])  # fmt: skip
        err = (pred - val_targets).abs().mean(0).numpy()
        print(f"epoch {epoch}: train loss {loss.item():.4f}, val abs error per dim {np.round(err, 4)}", flush=True)
    torch.save(model.state_dict(), args.ckpt)


def ndtw(p: np.ndarray, q: np.ndarray) -> float:
    cost = np.linalg.norm(p[:, None] - q[None], axis=2)
    acc = np.full((len(p) + 1, len(q) + 1), np.inf)
    acc[0, 0] = 0
    for i in range(1, len(p) + 1):
        for j in range(1, len(q) + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(np.exp(-acc[-1, -1] / (len(q) * D_TH)))


def score(args) -> None:
    from diffusers.models import AutoencoderKL

    device = "cuda"
    if args.backbone == "latent":
        vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae").to(device)
        model = IDM().to(device).eval()
    else:
        model = DinoIDM().to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    rows = []
    for name in args.models:
        for path in sorted(glob.glob(os.path.join(args.gen, name, "*.mp4"))):
            clip = os.path.basename(path)[:-4]
            states = np.load(os.path.join(args.gen, "states", f"{clip}.npz"))
            state, gripper = states["state"], states["gripper"]
            video = np.stack(imageio.mimread(path))
            with torch.no_grad():
                if args.backbone == "latent":
                    frames = torch.from_numpy(video).permute(0, 3, 1, 2).float().to(device)
                    latent = vae.encode(frames / 127.5 - 1).latent_dist.mean * vae.config.scaling_factor
                    pred = model(torch.cat([latent[:-1], latent[1:]], dim=1))
                else:
                    frames = resize(video).to(device)
                    pred = model(torch.cat([frames[:-1], frames[1:]], dim=1))
            pred = pred.float().cpu().numpy()
            row, path_pred = trajectory_scores(pred, state, gripper)
            rows.append({"model": name, "clip": clip, **row})
            folder = os.path.join(args.gen, name, f"idm_{args.backbone}")
            os.makedirs(folder, exist_ok=True)
            np.savez(os.path.join(folder, f"{clip}.npz"), pred=pred, path=path_pred)
    write_rows(os.path.join(args.gen, f"metrics_idm_{args.backbone}.csv"), rows, args.models)


def trajectory_scores(pred: np.ndarray, state: np.ndarray, gripper: np.ndarray) -> tuple[dict, np.ndarray]:
    """3-D scores of recovered actions [15, 7] against a clip's recorded poses [16, 6]; also the recovered path."""
    given = actions_from_states(state, gripper)
    path_pred, path_real = integrate(state[0], pred), state[:, :3]
    error = np.linalg.norm(path_pred - path_real, axis=1)
    length = lambda p: np.linalg.norm(np.diff(p, axis=0), axis=1).sum()  # noqa: E731
    return {
        "pos_err_cm": 100 * error[1:].mean(), "final_err_cm": 100 * error[-1], "ndtw_3d": ndtw(path_pred, path_real),
        "path_ratio_3d": length(path_pred) / max(length(path_real), 1e-6),
        "action_err": float(np.abs(pred - given)[:, :6].mean()),
    }, path_pred  # fmt: skip


def write_rows(out: str, rows: list[dict], models: list[str]) -> None:
    """Replace the rows of the given models in a metrics CSV and print their means."""
    kept = []
    if os.path.exists(out):
        with open(out) as f:
            kept = [r for r in csv.DictReader(f) if r["model"] not in models]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "clip", *FIELDS])
        writer.writeheader()
        writer.writerows(kept + rows)
    for name in models:
        values = {k: np.array([r[k] for r in rows if r["model"] == name]) for k in FIELDS}
        print(f"{name:26s} clips {len(values['ndtw_3d']):3d}  "
              + "  ".join(f"{k} {v.mean():.3f}" for k, v in values.items()))  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("train", "score"))
    parser.add_argument("--data", help="train: folder holding opensource_robotdata/")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--backbone", choices=("latent", "dinov2"), default="latent")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024, help="1024 for latent; use 128 for dinov2")
    parser.add_argument("--episodes", type=int, default=2000, help="dinov2: episodes held in memory")
    parser.add_argument("--gen", help="score: folder with states/ and the model folders")
    parser.add_argument("--models", nargs="+", help="score: model folders, with real and still as controls")
    args = parser.parse_args()
    (train if args.action == "train" else score)(args)


if __name__ == "__main__":
    main()
