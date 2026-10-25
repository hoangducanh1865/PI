"""
The packed Grasp-Anything++ splits (scripts/prepare_data.py) as a PyTorch dataset and a Lightning data module.

A sample is one prompt: the 224 x 224 image (scaled to [0, 1], minus its own mean, as in GR-ConvNet), the four target
maps of GR-ConvNet (Q, cos 2 theta, sin 2 theta, opening / WIDTH_SCALE) and, when a text model is given, the cached
CLIP text features (scripts/cache_text_emb.py): token sequence, its mask and the pooled embedding. Everything except
the maps is loaded into RAM once (train: about 1.3 GB), so the data can be read straight from Google Drive.

Author: Duc Anh Hoang
"""

import os

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.grasp import WIDTH_SCALE, draw_maps


class GraspAnything(Dataset):
    def __init__(self, data_dir, split, text_model=None):
        with h5py.File(os.path.join(data_dir, f"{split}.h5")) as f:
            self.images = f["images"][:]
            self.image_index = f["image_index"][:]
            self.all_grasps = f["grasps"][:]
            self.offsets = f["grasp_offsets"][:]
            self.keys = [k.decode() for k in f["keys"][:]]
            self.prompts = [p.decode() for p in f["prompts"][:]]
        self.size = self.images.shape[1]
        self.text = None
        if text_model:
            cache = dict(np.load(os.path.join(data_dir, f"{split}_text_{text_model}.npz")))
            if list(cache["keys"]) != self.keys:
                raise ValueError(f"{split}_text_{text_model}.npz does not match {split}.h5: rerun cache_text_emb.py")
            self.text = cache

    def __len__(self):
        return len(self.keys)

    def grasps(self, i):
        """Ground-truth rows (quality, x, y, w, h, theta) of sample i, in pixels of the 224 x 224 image."""
        return self.all_grasps[self.offsets[i] : self.offsets[i + 1]]

    def __getitem__(self, i):
        image = self.images[self.image_index[i]].astype(np.float32) / 255.0
        image -= image.mean()
        q, angle, width = draw_maps(self.grasps(i), self.size)
        maps = np.stack([q, np.cos(2 * angle), np.sin(2 * angle), np.clip(width / WIDTH_SCALE, 0.0, 1.0)])
        sample = {"image": torch.from_numpy(image.transpose(2, 0, 1)), "maps": torch.from_numpy(maps), "index": i}
        if self.text is not None:
            t = self.text["text_index"][i]
            sample["text_tokens"] = torch.from_numpy(self.text["tokens"][t].astype(np.float32))
            sample["text_mask"] = torch.from_numpy(self.text["token_mask"][t])
            sample["text_emb"] = torch.from_numpy(self.text["emb"][i])
        return sample


class GraspAnythingData(L.LightningDataModule):
    """Train on train.h5, validate on test_seen.h5 (the authors keep the last 10% of the Seen objects for this)."""

    def __init__(self, dir, text_model=None, batch_size=8, num_workers=2):
        super().__init__()
        self.dir, self.text_model, self.batch_size, self.num_workers = dir, text_model, batch_size, num_workers

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_set = GraspAnything(self.dir, "train", self.text_model)
        self.val_set = GraspAnything(self.dir, "test_seen", self.text_model)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_set, 32, num_workers=self.num_workers, persistent_workers=self.num_workers > 0)
