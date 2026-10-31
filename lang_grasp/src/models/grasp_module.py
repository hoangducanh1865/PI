"""
Training and validation of the grasp network with Lightning.

Loss, optimiser and validation follow the Grasp-Anything authors' train_network.py and grasp_model.py: the sum of the
smooth L1 losses of the four maps, Adam (learning rate 1e-3), and the share of validation samples whose decoded grasp
succeeds (IoU > 25%, angle < 30 degrees), which also picks the best checkpoint. With contrast_weight > 0 the
proposed region-text contrastive loss (src/models/region_text_contrast.py) is added, weighted by contrast_weight; its
projection heads are used in training only.

Author: Duc Anh Hoang
"""

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.grconvnet import GenerativeResnet
from src.models.region_text_contrast import RegionTextContrast
from src.utils.grasp import decode, success

MAPS = ("q", "cos", "sin", "width")


class GraspModule(L.LightningModule):
    def __init__(
        self,
        fusion=None,
        channel_size=32,
        dropout=0.1,
        lr=1e-3,
        text_model=None,
        contrast_weight=0.0,
        contrast_temperature=0.07,
    ):
        super().__init__()
        self.save_hyperparameters()  # text_model is kept so that eval.py reads the right text cache
        self.net = GenerativeResnet(3, channel_size, dropout, fusion)
        self.contrast = (
            RegionTextContrast(channel_size * 4, temperature=contrast_temperature) if contrast_weight else None
        )
        self.val_hits = []

    def forward(self, batch, return_features=False):
        return self.net(batch["image"], batch.get("text_tokens"), batch.get("text_mask"), return_features)

    def training_step(self, batch, _):
        if self.contrast is None:
            pred = self(batch)
        else:
            pred, features = self(batch, return_features=True)
        losses = {name: F.smooth_l1_loss(pred[:, i], batch["maps"][:, i]) for i, name in enumerate(MAPS)}
        loss = sum(losses.values())
        if self.contrast is not None:
            losses["contrast"], accuracy = self.contrast(
                features, batch["regions"], batch["region_valid"], batch["text_emb"]
            )
            loss = loss + self.hparams.contrast_weight * losses["contrast"]
            self.log("train/contrast_acc", accuracy, on_step=False, on_epoch=True)
        self.log_dict({f"train/{name}_loss": v for name, v in losses.items()}, on_step=False, on_epoch=True)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        pred = self(batch)
        self.log("val/loss", sum(F.smooth_l1_loss(pred[:, i], batch["maps"][:, i]) for i in range(4)), on_epoch=True)
        dataset = self.trainer.val_dataloaders.dataset
        for maps, i in zip(pred.float().cpu().numpy(), batch["index"].tolist(), strict=True):
            self.val_hits.append(success(decode(*maps), dataset.grasps(i)))

    def on_validation_epoch_end(self):
        self.log("val/success", float(np.mean(self.val_hits)), prog_bar=True)
        self.val_hits = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


@torch.no_grad()
def predict(net, dataset, batch_size=32, num_workers=2, region=None):
    """The decoded grasp (x, y, w, h, theta), or None, of every sample of a dataset.

    region: optional function sample index -> bool map (or None) restricting where the grasp is taken (decode).
    """
    device = next(net.parameters()).device
    net.eval()
    grasps = []
    for batch in DataLoader(dataset, batch_size, num_workers=num_workers):
        tokens, mask = (batch[k].to(device) if k in batch else None for k in ("text_tokens", "text_mask"))
        maps = net(batch["image"].to(device), tokens, mask).float().cpu().numpy()
        start = len(grasps)
        grasps.extend(decode(*m, region=region(start + j) if region else None) for j, m in enumerate(maps))
    return grasps
