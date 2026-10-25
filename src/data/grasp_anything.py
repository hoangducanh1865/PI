"""
One split of the packed Grasp-Anything++ data (scripts/prepare_data.py) as a PyTorch dataset.

A sample is one prompt: the 224 x 224 image (scaled to [0, 1], minus its own mean, as in GR-ConvNet), the cached text
embedding (scripts/cache_text_emb.py; zeros when no text model is given) and the four target maps of GR-ConvNet
(Q, cos 2 theta, sin 2 theta, opening / WIDTH_SCALE). Everything except the maps is loaded into RAM once
(train: about 1.2 GB).

Author: Duc Anh Hoang
"""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

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
        if text_model:
            cache = np.load(os.path.join(data_dir, f"{split}_text_{text_model}.npz"))
            if list(cache["keys"]) != self.keys:
                raise ValueError(
                    f"{split}_text_{text_model}.npz does not match {split}.h5: run cache_text_emb.py again"
                )
            self.text = cache["emb"].astype(np.float32)
        else:
            self.text = np.zeros((len(self.keys), 1), dtype=np.float32)

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
        return {
            "image": torch.from_numpy(image.transpose(2, 0, 1)),
            "text": torch.from_numpy(self.text[i]),
            "maps": torch.from_numpy(maps),
            "index": i,
        }
