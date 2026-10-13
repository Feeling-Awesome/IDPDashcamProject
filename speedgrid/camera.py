"""Camera model and flat-ground (inverse perspective) geometry.

The dashcam is modelled as a pinhole camera mounted at height ``h`` above a
flat road.  Its orientation (pitch / yaw) is recovered from the road's
vanishing point, so the image <-> ground-plane mapping is a homography.
Everything downstream works in a metric bird's-eye view (BEV) of the road,
where 1 metre is a fixed number of pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CameraModel:
    width: int
    height: int
    hfov_deg: float = 100.0          # horizontal field of view of the dashcam
    cam_height_m: float = 1.3        # lens height above the road
    vp: tuple[float, float] | None = None  # vanishing point (u, v) in pixels
    bonnet_frac: float = 0.0         # bottom fraction of the frame hidden by the bonnet

    # BEV layout
    bev_half_width_m: float = 14.0
    bev_length_m: float = 36.0
    ppm: float = 14.0                # BEV pixels per metre

    # derived
    K: np.ndarray = field(init=False, repr=False)
    G2I: np.ndarray = field(init=False, repr=False)   # ground (X,Z,1) -> image
    BEV2I: np.ndarray = field(init=False, repr=False)  # BEV pixel -> image pixel
    I2BEV: np.ndarray = field(init=False, repr=False)
    z_min: float = field(init=False)

    def __post_init__(self):
        self.rebuild()

    # ------------------------------------------------------------------ setup
    @property
    def focal(self) -> float:
        return (self.width / 2.0) / np.tan(np.radians(self.hfov_deg) / 2.0)

    def rebuild(self):
        f = self.focal
        cx, cy = self.width / 2.0, self.height / 2.0
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        vp = self.vp if self.vp is not None else (cx, cy)

        # Road forward direction in camera coordinates = back-projected VP.
        d = np.linalg.inv(self.K) @ np.array([vp[0], vp[1], 1.0])
        d /= np.linalg.norm(d)
        # World "down" = camera y-axis made orthogonal to d (no roll assumed).
        ey = np.array([0.0, 1.0, 0.0])
        yw = ey - ey.dot(d) * d
        yw /= np.linalg.norm(yw)
        xw = np.cross(yw, d)
        # Ground point (X, h, Z) -> camera = X*xw + h*yw + Z*d
        self.G2I = self.K @ np.column_stack([xw, d, self.cam_height_m * yw])
        self._axes = (xw, yw, d)

        # Nearest visible ground distance (bottom of usable image, centre column)
        v_bottom = self.height * (1.0 - self.bonnet_frac) - 2
        X, Z = self.image_to_ground(np.array([[cx, v_bottom]]))[0]
        self.z_min = float(max(Z, 0.5))

        # BEV pixel (bx, by) -> ground: X = bx/ppm - half, Z = z_max - by/ppm
        z_max = self.z_min + self.bev_length_m
        B2G = np.array([[1.0 / self.ppm, 0, -self.bev_half_width_m],
                        [0, -1.0 / self.ppm, z_max],
                        [0, 0, 1.0]])
        self.BEV2I = self.G2I @ B2G
        self.I2BEV = np.linalg.inv(self.BEV2I)
        self._valid = None
        self._inner_valid = None

    # ------------------------------------------------------------- transforms
    @property
    def bev_size(self) -> tuple[int, int]:
        return (int(round(2 * self.bev_half_width_m * self.ppm)),
                int(round(self.bev_length_m * self.ppm)))

    def image_to_ground(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return np.zeros((0, 2))
        p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64),
                                     np.linalg.inv(self.G2I))
        return p.reshape(-1, 2)

    def project_3d(self, X, up, Z) -> np.ndarray:
        """Project road-frame points (lateral X, height above road, forward Z) to pixels."""
        xw, yw, d = self._axes
        P = (np.outer(X, xw) + np.outer(self.cam_height_m - np.asarray(up, float), yw)
             + np.outer(Z, d))
        p = P @ self.K.T
        return p[:, :2] / p[:, 2:3]

    def bev_to_image(self, pts: np.ndarray) -> np.ndarray:
        p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), self.BEV2I)
        return p.reshape(-1, 2)

    def warp_to_bev(self, img: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
        return cv2.warpPerspective(img, self.I2BEV, self.bev_size, flags=interp,
                                   borderMode=cv2.BORDER_CONSTANT)

    def warp_to_image(self, bev: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
        return cv2.warpPerspective(bev, self.BEV2I, (self.width, self.height), flags=interp)

    def valid_bev_mask(self) -> np.ndarray:
        """BEV pixels that actually see the road (inside the image, below bonnet)."""
        if self._valid is None:
            m = np.zeros((self.height, self.width), np.uint8)
            m[: int(self.height * (1.0 - self.bonnet_frac)) - 2, :] = 255
            self._valid = self.warp_to_bev(m, cv2.INTER_NEAREST)
        return self._valid

    def scale_height(self, factor: float):
        """Uniformly rescale the metric world (used by lane-width auto calibration)."""
        self.cam_height_m *= factor
        self.rebuild()


# --------------------------------------------------------------------------
# Vanishing point estimation
# --------------------------------------------------------------------------

def _line_segments(img: np.ndarray, roi_top: int):
    """Line segments from edges of brightness *and* colour (grass/asphalt borders
    are often low-contrast in grey but obvious in saturation), with thresholds
    adapted to each frame's contrast."""
    if img.ndim == 2:
        chans = [img]
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        chans = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), hsv[:, :, 1]]
    edges = None
    for ch in chans:
        ch = cv2.GaussianBlur(ch, (5, 5), 0)
        sub = ch[roi_top:]
        gx = cv2.Sobel(sub, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(sub, cv2.CV_32F, 0, 1)
        mag = np.hypot(gx, gy)
        hi = float(np.percentile(mag, 92))
        hi = max(hi, 20.0)
        e = cv2.Canny(ch, 0.4 * hi, hi)
        edges = e if edges is None else (edges | e)
    edges[:roi_top] = 0
    return cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                           minLineLength=img.shape[1] // 30, maxLineGap=20)


def _paint_segments(img: np.ndarray, roi_top: int):
    """Line segments of painted lane lines (thin bright stripes): a white top-hat keeps
    paint and drops trees, verges, clouds and text, which only distract the estimate."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = gray.shape
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(W // 60, 5) | 1, 1))
    th = cv2.morphologyEx(cv2.GaussianBlur(gray, (3, 3), 0), cv2.MORPH_TOPHAT, k)
    sub = th[roi_top:]
    thr = max(25.0, float(np.percentile(sub, 98.5)))
    bw = (th > thr).astype(np.uint8) * 255
    bw[:roi_top] = 0
    return cv2.HoughLinesP(bw, 1, np.pi / 180, threshold=20, minLineLength=W // 25, maxLineGap=25)


def estimate_vanishing_point(frames: list[np.ndarray]) -> tuple[float, float] | None:
    """Robust VP from the converging road/lane lines across sampled frames.

    Hough line segments (painted lines weighted x3, plus general edges) on the
    lower part of the frame are split into left- and right-leaning sets; the
    intersections of left/right pairs vote in a 2-D histogram and the peak wins.
    """
    if not frames:
        return None
    H, W = frames[0].shape[:2]
    pts = []
    for fr in frames:
        sources = []
        ps = _paint_segments(fr, int(H * 0.45))
        if ps is not None:
            sources.append((ps, 3.0))          # painted lines count three times as much
        es = _line_segments(fr, int(H * 0.45))
        if es is not None:
            sources.append((es, 1.0))
        if not sources:
            continue
        left, right = [], []
        for segs, wgt in sources:
            for x1, y1, x2, y2 in segs.reshape(-1, 4).astype(float):
                if x2 == x1:
                    continue
                ang = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180) % 180
                length = np.hypot(x2 - x1, y2 - y1) * wgt
                line = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1])  # homogeneous line
                mx = (x1 + x2) / 2
                if 10 < ang < 80 and mx > W * 0.35:        # "\" on the right side
                    right.append((line, length))
                elif 100 < ang < 170 and mx < W * 0.65:    # "/" on the left side
                    left.append((line, length))
        left = sorted(left, key=lambda t: -t[1])[:20]
        right = sorted(right, key=lambda t: -t[1])[:20]
        for la, wa in left:
            for lb, wb in right:
                p = (la[1] * lb[2] - la[2] * lb[1], la[2] * lb[0] - la[0] * lb[2],
                     la[0] * lb[1] - la[1] * lb[0])
                if abs(p[2]) < 1e-9:
                    continue
                x, y = p[0] / p[2], p[1] / p[2]
                if 0.1 * W < x < 0.9 * W and 0.1 * H < y < 0.92 * H:
                    pts.append((x, y, np.sqrt(wa * wb)))
    if len(pts) < 8:
        return None
    pts = np.array(pts)
    # 2D vote: the true VP is where many independent pairs agree (a mode, not a mean)
    cell = 4
    acc = np.zeros((H // cell + 1, W // cell + 1), np.float32)
    np.add.at(acc, ((pts[:, 1] / cell).astype(int), (pts[:, 0] / cell).astype(int)), pts[:, 2])
    acc = cv2.GaussianBlur(acc, (0, 0), 2.0)
    py, px = np.unravel_index(int(np.argmax(acc)), acc.shape)
    c = np.array([(px + 0.5) * cell, (py + 0.5) * cell])
    near = np.linalg.norm(pts[:, :2] - c, axis=1) < max(W * 0.025, 8)
    if near.sum() >= 3:
        c = np.average(pts[near, :2], axis=0, weights=pts[near, 2])
    return float(c[0]), float(c[1])
