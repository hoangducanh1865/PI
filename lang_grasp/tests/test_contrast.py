"""
src/models/region_text_contrast.py: the loss prefers the prompted region, ignores invalid slots, and skips samples
without negatives; the network returns its 56 x 56 encoder features for it.

Author: Duc Anh Hoang
"""

import pytest

torch = pytest.importorskip("torch")
from src.models.grconvnet import GenerativeResnet  # noqa: E402
from src.models.region_text_contrast import RegionTextContrast  # noqa: E402


def regions_and_feats():
    """Two regions (left / right half of a 4 x 4 map) whose features are one-hot vectors e0 and e1."""
    regions = torch.zeros(1, 3, 4, 4)
    regions[0, 0, :, :2] = 1  # slot 0: the prompted object, left half
    regions[0, 1, :, 2:] = 1  # slot 1: another object, right half
    feats = torch.zeros(1, 4, 4, 4)
    feats[0, 0, :, :2] = 1
    feats[0, 1, :, 2:] = 1
    valid = torch.tensor([[True, True, False]])
    return feats, regions, valid


def test_prefers_the_prompted_region():
    torch.manual_seed(0)
    loss = RegionTextContrast(feat_dim=4, text_dim=4, dim=4)
    with torch.no_grad():  # identity projections: region k's embedding is e_k
        for proj in (loss.region_proj, loss.text_proj):
            proj.weight.copy_(torch.eye(4))
            proj.bias.zero_()
    feats, regions, valid = regions_and_feats()
    right, acc_right = loss(feats, regions, valid, torch.tensor([[1.0, 0, 0, 0]]))  # text matches slot 0
    wrong, acc_wrong = loss(feats, regions, valid, torch.tensor([[0, 1.0, 0, 0]]))  # text matches slot 1
    assert right < 1e-3 < wrong
    assert acc_right == 1 and acc_wrong == 0


def test_invalid_slots_and_no_negative():
    loss = RegionTextContrast(feat_dim=4, text_dim=4, dim=4)
    feats, regions, valid = regions_and_feats()
    text = torch.randn(1, 4)
    base, _ = loss(feats, regions, valid, text)
    regions[0, 2] = 5.0  # an invalid slot must not change the loss
    assert torch.allclose(base, loss(feats, regions, valid, text)[0])
    only_target = torch.tensor([[True, False, False]])
    assert loss(feats, regions, only_target, text)[0] == 0  # no negative: skipped


def test_network_returns_encoder_features():
    net = GenerativeResnet(fusion="albef").eval()
    maps, feats = net(torch.randn(1, 3, 224, 224), torch.randn(1, 5, 512), torch.ones(1, 5, dtype=torch.bool), True)
    assert maps.shape == (1, 4, 224, 224) and feats.shape == (1, 128, 56, 56)
