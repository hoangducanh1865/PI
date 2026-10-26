"""
The packed Grasp-Anything++ splits (scripts/prepare_data.py) as a PyTorch dataset and a Lightning data module.

A sample is one prompt: the 224 x 224 image (scaled to [0, 1], minus its own mean, as in GR-ConvNet), the four target
maps of GR-ConvNet (Q, cos 2 theta, sin 2 theta, opening / WIDTH_SCALE) and, when a text model is given, the cached
CLIP text features (scripts/cache_text_emb.py): token sequence, its mask and the pooled embedding. Everything except
the maps is loaded into RAM once (train: about 1.3 GB), so the data can be read straight from Google Drive.

With regions=True (training of the proposed contrastive loss) a sample also holds the pooling weights of the scene's
regions at the encoder resolution (56 x 56, the share of each 4 x 4 cell inside the mask): slot 0 the prompted object
(the union of its part masks), then the other objects of the same image in this split, then (last slot) the rest of the
scene outside every annotated object; region_valid marks the used slots. Only this split's masks are used, so no test
annotation enters training.

Author: Duc Anh Hoang
"""

import itertools
import os

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.grasp import WIDTH_SCALE, draw_maps


class GraspAnything(Dataset):
    def __init__(self, data_dir, split, text_model=None, regions=False):
        with h5py.File(os.path.join(data_dir, f"{split}.h5")) as f:
            self.images = f["images"][:]
            self.image_index = f["image_index"][:]
            self.all_grasps = f["grasps"][:]
            self.offsets = f["grasp_offsets"][:]
            self.keys = [k.decode() for k in f["keys"][:]]
            self.prompts = [p.decode() for p in f["prompts"][:]]
            packed_masks = f["masks"][:] if regions else None
        self.size = self.images.shape[1]
        self.regions = regions
        if regions:
            self._build_regions(packed_masks)
        self.text = None
        if text_model:
            cache = dict(np.load(os.path.join(data_dir, f"{split}_text_{text_model}.npz")))
            if list(cache["keys"]) != self.keys:
                raise ValueError(f"{split}_text_{text_model}.npz does not match {split}.h5: rerun cache_text_emb.py")
            self.text = cache

    def _build_regions(self, packed_masks, cell=4):
        objects = [k.rsplit("_", 1)[0] for k in self.keys]  # <image>_<object>_<part> -> <image>_<object>
        self.object_index = np.zeros(len(self.keys), dtype=np.int64)
        pooled, image_objects = [], {}
        side = self.size // cell
        for o, rows in itertools.groupby(range(len(self.keys)), key=lambda i: objects[i]):  # keys are sorted
            rows = list(rows)
            union = np.zeros((self.size, self.size), dtype=bool)
            for i in rows:
                union |= np.unpackbits(packed_masks[i]).reshape(self.size, self.size).astype(bool)
            self.object_index[rows] = len(pooled)
            image_objects.setdefault(o.split("_")[0], []).append(len(pooled))
            pooled.append(union.reshape(side, cell, side, cell).mean((1, 3)).astype(np.float16))
        self.object_masks = np.stack(pooled)
        self.image_objects = image_objects
        self.slots = max(len(v) for v in image_objects.values()) + 1  # all objects of the busiest image + the rest

    def _regions(self, i):
        side = self.object_masks.shape[1]
        regions = np.zeros((self.slots, side, side), dtype=np.float32)
        target = self.object_index[i]
        scene = self.image_objects[self.keys[i].split("_")[0]]
        order = [target] + [j for j in scene if j != target]
        regions[: len(order)] = self.object_masks[order]
        regions[-1] = 1.0 - np.clip(self.object_masks[scene].astype(np.float32).sum(0), 0.0, 1.0)
        valid = regions.sum((1, 2)) >= 1.0  # at least one cell's worth of pixels
        valid[len(order) : -1] = False
        return torch.from_numpy(regions), torch.from_numpy(valid)

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
        if self.regions:
            sample["regions"], sample["region_valid"] = self._regions(i)
        return sample


class GraspAnythingData(L.LightningDataModule):
    """Train on train.h5, validate on test_seen.h5 (the authors keep the last 10% of the Seen objects for this)."""

    def __init__(self, dir, text_model=None, batch_size=8, num_workers=2, regions=False):
        super().__init__()
        self.dir, self.text_model, self.batch_size, self.num_workers = dir, text_model, batch_size, num_workers
        self.regions = regions  # region masks for the proposed contrastive loss (training set only)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_set = GraspAnything(self.dir, "train", self.text_model, self.regions)
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
