"""
src/models/grconvnet.py: the authors' GR-ConvNet3 size, the 2.07M parameters of the paper's "GR-ConvNet + CLIP"
(Table 9), output shapes, and that padded text tokens are ignored.

Author: Duc Anh Hoang
"""

import pytest

torch = pytest.importorskip("torch")
from src.models.grconvnet import GenerativeResnet  # noqa: E402


def count(net):
    return sum(p.numel() for p in net.parameters())


def test_sizes():
    assert count(GenerativeResnet()) == 1_898_308  # the authors' grconvnet3 with RGB input
    assert abs(count(GenerativeResnet(fusion="albef")) - 2_070_000) / 2_070_000 < 0.01


def test_forward_and_padding():
    torch.manual_seed(0)
    net = GenerativeResnet(fusion="albef").eval()
    image = torch.randn(2, 3, 224, 224)
    tokens, mask = torch.randn(2, 6, 512), torch.ones(2, 6, dtype=torch.bool)
    mask[1, 4:] = False
    out = net(image, tokens, mask)
    assert out.shape == (2, 4, 224, 224)
    tokens2 = tokens.clone()
    tokens2[1, 4:] = 100.0  # changing padded tokens must not change the output
    assert torch.allclose(out[1], net(image, tokens2, mask)[1], atol=1e-5)
    assert GenerativeResnet().eval()(image).shape == (2, 4, 224, 224)
