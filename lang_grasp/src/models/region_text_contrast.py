"""
Proposed method: region-text contrastive grounding loss, used only during training.

Motivation (our error analysis of the reproduced GR-ConvNet + CLIP baseline): about 56% of its failures grasp another
object than the one in the prompt. The baseline's grasp loss only says where to grasp; nothing tells the image encoder,
trained from scratch, which object a prompt names. This loss does: the encoder features of the prompted object, pooled
over its mask, must match the prompt's CLIP embedding better than the features of the other objects of the same scene
(hard negatives) and of the rest of the scene. Masks are needed only here, so inference costs nothing extra.

    r_k = sum_p w_k(p) f(p) / sum_p w_k(p)             region k: encoder features f (56 x 56) pooled with weights w_k
    s_k = cos(P_r r_k, P_t t) / tau                    P_r, P_t: linear projections; t: the prompt's CLIP embedding
    L   = -log( exp(s_0) / sum_k exp(s_k) )            k = 0 the prompted object, k > 0 the negatives

Author: Duc Anh Hoang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionTextContrast(nn.Module):
    def __init__(self, feat_dim=128, text_dim=512, dim=128, temperature=0.07):
        super().__init__()
        self.region_proj = nn.Linear(feat_dim, dim)
        self.text_proj = nn.Linear(text_dim, dim)
        self.temperature = temperature

    def forward(self, feats, regions, valid, text):
        """feats (B, C, H, W); regions (B, R, H, W) pooling weights, slot 0 = the prompted object; valid (B, R) bool;
        text (B, text_dim). Returns the loss and the share of samples whose prompted object scores highest."""
        weights = regions / regions.sum((2, 3), keepdim=True).clamp(min=1e-6)
        pooled = torch.einsum("bchw,brhw->brc", feats.float(), weights)
        z_region = F.normalize(self.region_proj(pooled), dim=-1)
        z_text = F.normalize(self.text_proj(text.float()), dim=-1)
        logits = torch.einsum("brd,bd->br", z_region, z_text) / self.temperature
        logits = logits.masked_fill(~valid, float("-inf"))
        used = valid[:, 0] & valid[:, 1:].any(1)  # a target and at least one negative
        if not used.any():
            return logits.new_zeros(()), logits.new_zeros(())
        logits = logits[used]
        target = logits.new_zeros(len(logits), dtype=torch.long)
        return F.cross_entropy(logits, target), (logits.argmax(1) == 0).float().mean()
