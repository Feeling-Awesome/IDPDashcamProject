"""3D vehicle boxes from one camera: a learned size/orientation model + geometry.

The network (trained by training/train_vehicle3d.py on the KITTI dataset) looks
at the image of one detected vehicle and predicts its real size (height, width,
length in metres) and its *observation angle* (which side of it we see).  It
never predicts distance - that depends on the camera, and would not transfer
from KITTI's camera to a dashcam.

Distance then comes from geometry ("Deep3DBox", Mousavian et al. 2017): place a
box of that size and heading in front of the camera, project its 8 corners, and
move it until its outline matches the detected 2D box.  The box's height in the
picture fixes the distance; its height above the road is left free, so neither
the horizon nor a flat road are assumed.  Sides of the 2D box cut by the frame
edge are simply not used.

Coordinates follow KITTI: a camera frame levelled with the road, x right, y
down, z forward; a vehicle's location is the centre of its bottom face;
rotation ry about y (ry = 0: length along x), observation angle
alpha = ry - atan2(x, z).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_ONNX = MODEL_DIR / "vehicle3d.onnx"
MODEL_JSON = MODEL_DIR / "vehicle3d.json"


# ------------------------------------------------------------------------ geometry
def box_corners(dims, loc, ry):
    """8 corners (3, 8) of a 3D box: dims (h, w, l), loc = bottom-centre (x, y, z)."""
    h, w, l = dims
    x = np.array([l, l, -l, -l, l, l, -l, -l]) / 2
    y = np.array([0, 0, 0, 0, -h, -h, -h, -h], float)
    z = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
    c, s = np.cos(ry), np.sin(ry)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return R @ np.vstack([x, y, z]) + np.asarray(loc, float).reshape(3, 1)


def project(P, cam):
    """Project level-frame points (3, N) with cam = (f, cx, cy, Rl) where Rl maps the
    level frame into the real (possibly pitched / yawed) camera frame."""
    f, cx, cy, Rl = cam
    Q = Rl @ P
    z = np.maximum(Q[2], 1e-3)
    return np.vstack([f * Q[0] / z + cx, f * Q[1] / z + cy])


def box2d(dims, loc, alpha, cam):
    """Outline (x1, y1, x2, y2) of the projected 3D box; ry follows from alpha and the ray."""
    ry = alpha + np.arctan2(loc[0], loc[2])
    p = project(box_corners(dims, loc, ry), cam)
    return np.array([p[0].min(), p[1].min(), p[0].max(), p[1].max()])


def solve_location(box, dims, alpha, cam, valid=(True, True, True, True), iters=12):
    """Bottom-centre location (x, y, z) in the level frame whose projected outline best
    matches the 2D box.  ``valid`` marks which box sides are real (not cut by the frame).
    Returns (loc, rms residual in px) or (None, inf)."""
    f, cx, cy, Rl = cam
    box = np.asarray(box, float)
    m = np.asarray(valid, bool)
    if m.sum() < 3 or not (m[1] and m[3]):
        return None, np.inf                       # need top and bottom for the distance
    h = dims[0]
    hp = max(box[3] - box[1], 1.0)
    z0 = f * h / hp
    # start on the ray through the box centre at that distance
    ray = np.linalg.solve(Rl, np.array([((box[0] + box[2]) / 2 - cx) / f, ((box[1] + box[3]) / 2 - cy) / f, 1.0]))
    loc = ray / ray[2] * z0
    loc[1] += h / 2                               # centre of the box -> its bottom
    scale = max(box[2] - box[0], box[3] - box[1], 1.0)
    for _ in range(iters):
        r0 = (box2d(dims, loc, alpha, cam) - box)[m] / scale
        J = np.zeros((m.sum(), 3))
        for k in range(3):
            d = np.zeros(3)
            d[k] = 1e-3 * max(loc[2], 1.0)
            J[:, k] = ((box2d(dims, loc + d, alpha, cam) - box)[m] / scale - r0) / d[k]
        step, *_ = np.linalg.lstsq(J, -r0, rcond=None)
        lim = 0.3 * max(loc[2], 1.0)              # damp big jumps
        n = np.linalg.norm(step)
        if n > lim:
            step *= lim / n
        loc = loc + step
        if loc[2] < 0.5:
            return None, np.inf
        if np.linalg.norm(step) < 1e-4 * loc[2]:
            break
    res = (box2d(dims, loc, alpha, cam) - box)[m]
    return loc, float(np.sqrt(np.mean(res ** 2)))


# --------------------------------------------------------------------- the model
def encode_angle(alpha):
    """Front and back of a box look the same in outline: learn the doubled angle."""
    return np.stack([np.sin(2 * alpha), np.cos(2 * alpha)], -1)


def decode_angle(sc):
    return np.arctan2(sc[..., 0], sc[..., 1]) / 2


def crop_square(img, box, size, pad=0.10, fill=(114, 114, 114)):
    """The vehicle with a small margin, padded (not stretched) to a square, resized."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * (1 + 2 * pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a = np.array([[size / side, 0, size / 2 - cx * size / side],
                  [0, size / side, size / 2 - cy * size / side]], np.float32)
    return cv2.warpAffine(img, a, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=fill)


class Vehicle3DModel:
    """The trained size/orientation network (ONNX, run with OpenCV - no PyTorch needed)."""

    def __init__(self, onnx_path=MODEL_ONNX, json_path=MODEL_JSON):
        self.meta = json.loads(Path(json_path).read_text())
        self.net = cv2.dnn.readNet(str(onnx_path))
        self.size = int(self.meta["input_size"])
        self.mean = np.array(self.meta["mean_rgb"], np.float32)
        self.std = np.array(self.meta["std_rgb"], np.float32)
        self.dims_mean = np.array(self.meta["dims_mean"], np.float32)       # log-space offset
        self.depth_sd = float(self.meta.get("depth_rel_sd", 0.12))          # from KITTI validation

    @staticmethod
    def available() -> bool:
        return MODEL_ONNX.exists() and MODEL_JSON.exists()

    def predict(self, img_bgr, boxes):
        """For each box: (dims (h, w, l) in m, observation angle alpha)."""
        if not len(boxes):
            return []
        crops = [crop_square(img_bgr, b, self.size)[:, :, ::-1].astype(np.float32) / 255.0 for b in boxes]
        x = (np.stack(crops) - self.mean) / self.std
        self.net.setInput(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
        out = self.net.forward()                                              # (N, 5)
        dims = np.exp(out[:, :3] + self.dims_mean)
        sc = out[:, 3:5] / np.maximum(np.linalg.norm(out[:, 3:5], axis=1, keepdims=True), 1e-6)
        return [(dims[i], float(decode_angle(sc[i]))) for i in range(len(boxes))]
