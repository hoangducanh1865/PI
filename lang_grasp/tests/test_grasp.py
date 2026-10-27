"""
src/utils/grasp.py: rectangle geometry, IoU and angle rule, and the round trip ground truth -> maps -> decoded grasp.

Author: Duc Anh Hoang
"""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("skimage")
from src.utils.grasp import WIDTH_SCALE, corners, decode, draw_maps, iou, success  # noqa: E402

G = np.array([112.0, 112.0, 60.0, 20.0, 30.0])  # x, y, w, h, theta


def test_corners_follow_the_opening_direction():
    pts = corners(0.0, 0.0, 2.0, 0.0, 90.0)  # opening along (cos 90, sin 90) = straight down the image
    np.testing.assert_allclose(pts[1], [0.0, 1.0], atol=1e-9)


def test_iou_rules():
    assert iou(G, G) == pytest.approx(1.0)
    assert iou(G, G + [0, 0, 0, 0, 180]) == pytest.approx(1.0)  # 180 degrees apart is the same grasp
    assert iou(G, G + [0, 0, 0, 0, 31]) == 0.0  # angle rule
    assert iou(G, G + [200, 0, 0, 0, 0]) == 0.0  # no overlap
    assert 0.25 < iou(G, G + [10, 0, 0, 0, 0]) < 1.0


def test_success_and_round_trip():
    gts = np.array([[1.0, *G]])
    assert success(G, gts) and not success(None, gts)
    q, angle, width = draw_maps(gts, 224)
    pred = decode(q, np.cos(2 * angle), np.sin(2 * angle), np.clip(width / WIDTH_SCALE, 0, 1))
    np.testing.assert_allclose(pred[[0, 1, 2, 4]], G[[0, 1, 2, 4]], atol=1.5)
    assert success(pred, gts)
