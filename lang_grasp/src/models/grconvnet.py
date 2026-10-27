"""
GR-ConvNet (Kumra et al., IROS 2020), optionally conditioned on the prompt through an ALBEF-style fusion layer.

GenerativeResnet and ResidualBlock are adapted from the Grasp-Anything authors' code (github.com/Fsoft-AIC/
Grasp-Anything, inference/models/grconvnet3.py and grasp_model.py, MIT licence), which is the network of their
"GR-ConvNet + CLIP" baseline: same layers, initialisation and head dropout (1,898,308 parameters for RGB input);
only the text fusion is added. The paper (Vuong et al., CVPR 2024, App. E.1) fuses the
CLIP text embedding and the visual features with ALBEF (Li et al., NeurIPS 2021) but does not say where. Here one
ALBEF layer (BERT-style, post-LayerNorm: cross-attention, then a feed-forward block) sits at the end of the encoder,
with the 56 x 56 image features as queries over the CLIP text tokens, so the decoder still gets a feature map; its
size makes the whole network 2.06M parameters, as the 2.07M of the paper's Table 9.

Author: Duc Anh Hoang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = self.bn2(self.conv2(x))
        return x + x_in


class AlbefFusion(nn.Module):
    """One ALBEF / BERT layer with cross-attention: image tokens attend to text tokens, then a feed-forward block."""

    def __init__(self, dim, text_dim=512, heads=4, ffn_dim=128, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, tokens, mask):
        """x: image features (B, C, H, W); tokens: (B, L, text_dim); mask: (B, L), True for real tokens."""
        b, c, h, w = x.shape
        q = x.flatten(2).transpose(1, 2)  # (B, H * W, C): one query per position
        kv = self.text_proj(tokens)
        q = self.norm1(q + self.attn(q, kv, kv, key_padding_mask=~mask, need_weights=False)[0])
        q = self.norm2(q + self.ffn(q))
        return q.transpose(1, 2).reshape(b, c, h, w)


class GenerativeResnet(nn.Module):
    """GR-ConvNet3: Q, cos 2 theta, sin 2 theta and width maps at the input resolution."""

    def __init__(self, input_channels=3, channel_size=32, dropout=0.1, fusion=None):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)
        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)
        self.res = nn.Sequential(*[ResidualBlock(channel_size * 4, channel_size * 4) for _ in range(5)])
        self.fusion = AlbefFusion(channel_size * 4) if fusion == "albef" else None
        self.conv4 = nn.ConvTranspose2d(
            channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1, output_padding=1
        )
        self.bn4 = nn.BatchNorm2d(channel_size * 2)
        self.conv5 = nn.ConvTranspose2d(
            channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2, output_padding=1
        )
        self.bn5 = nn.BatchNorm2d(channel_size)
        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)
        self.heads = nn.ModuleList([nn.Conv2d(channel_size, 1, kernel_size=2) for _ in range(4)])  # Q, cos, sin, width
        self.dropout = nn.Dropout(p=dropout)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, image, text_tokens=None, text_mask=None, return_features=False):
        x = F.relu(self.bn1(self.conv1(image)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res(x)
        features = x  # image-only encoder features (B, 4C, H / 4, W / 4), before the text fusion
        if self.fusion is not None:
            x = self.fusion(x, text_tokens, text_mask)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        maps = torch.cat([head(self.dropout(x)) for head in self.heads], dim=1)  # (B, 4, H, W)
        return (maps, features) if return_features else maps
