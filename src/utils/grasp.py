"""
Grasp rectangles: the four training maps, decoding a network output, and the success test.

A grasp is a row (x, y, w, h, theta): centre in pixels, opening w, finger width h, angle theta in degrees; with the
image y axis pointing down, the opening direction is (cos theta, sin theta). Rewritten from the GR-ConvNet code used by
the Grasp-Anything authors (github.com/Fsoft-AIC/Grasp-Anything, utils/dataset_processing/grasp.py, evaluation.py,
inference/post_process.py, MIT licence) for (x, y, w, h, theta) arrays, with the same rules:
    maps     Q = 1 on the middle third of each rectangle (w / 3 x h), angle and opening stored there
    decode   Gaussian smoothing (sigma 2 for Q and angle, 1 for width), the highest peak of Q above 0.2 at least
             20 px from the border, h = w / 2
    success  IoU > 25% with any ground-truth grasp whose angle differs by less than 30 degrees
One change: their evaluation multiplies the predicted width by 150 while their dataset divides it by 112
(output_size / 2); both sides use WIDTH_SCALE here.

Author: Duc Anh Hoang
"""

import numpy as np
from skimage.draw import polygon
from skimage.feature import peak_local_max
from skimage.filters import gaussian

WIDTH_SCALE = 112.0  # width map = opening / WIDTH_SCALE, clipped to [0, 1] (224 / 2, as in the authors' dataset)


def corners(x, y, w, h, theta_deg):
    """The 4 corners (x, y) of a grasp rectangle."""
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    dx = np.array([-w, w, w, -w]) / 2  # along the opening
    dy = np.array([-h, -h, h, h]) / 2  # along the fingers
    return np.stack([x + dx * c - dy * s, y + dx * s + dy * c], axis=1)


def rasterize(x, y, w, h, theta_deg, shape=None):
    """Pixels (rows, cols) inside the rectangle."""
    pts = corners(x, y, w, h, theta_deg)
    return polygon(pts[:, 1], pts[:, 0], shape)


def draw_maps(grasps, size):
    """Training maps from ground-truth rows (quality, x, y, w, h, theta): Q, angle (radians) and opening (pixels)."""
    q, angle, width = (np.zeros((size, size), dtype=np.float32) for _ in range(3))
    for _, x, y, w, h, theta in grasps:
        rr, cc = rasterize(x, y, w / 3, h, theta, (size, size))
        q[rr, cc] = 1.0
        angle[rr, cc] = np.deg2rad(theta)
        width[rr, cc] = w
    return q, angle, width


def decode(q, cos2, sin2, width):
    """The best grasp (x, y, w, h, theta) in network outputs (width map in [0, 1]), or None if Q has no peak."""
    q = gaussian(q, 2.0, preserve_range=True)
    angle = gaussian(np.arctan2(sin2, cos2) / 2.0, 2.0, preserve_range=True)
    width = gaussian(width * WIDTH_SCALE, 1.0, preserve_range=True)
    peaks = peak_local_max(q, min_distance=20, threshold_abs=0.2, num_peaks=1)
    if not len(peaks):
        return None
    r, c = peaks[0]
    return np.array([c, r, width[r, c], width[r, c] / 2, np.rad2deg(angle[r, c])])


def iou(a, b, max_angle_deg=30.0):
    """IoU of two grasps (x, y, w, h, theta), 0 if their angles differ by more than max_angle_deg (mod 180)."""
    if abs((a[4] - b[4] + 90.0) % 180.0 - 90.0) > max_angle_deg:
        return 0.0
    ra, ca = rasterize(*a)
    rb, cb = rasterize(*b)
    if not len(ra) or not len(rb):
        return 0.0
    r0, c0 = min(ra.min(), rb.min()), min(ca.min(), cb.min())
    canvas = np.zeros((max(ra.max(), rb.max()) - r0 + 1, max(ca.max(), cb.max()) - c0 + 1), dtype=np.uint8)
    canvas[ra - r0, ca - c0] += 1
    canvas[rb - r0, cb - c0] += 1
    return float((canvas == 2).sum() / (canvas > 0).sum())


def success(pred, gts, threshold=0.25):
    """True if IoU > threshold with any ground-truth grasp (rows: quality, x, y, w, h, theta)."""
    return pred is not None and any(iou(pred, g[1:]) > threshold for g in gts)
