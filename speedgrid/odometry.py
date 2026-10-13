"""Speed from the road surface itself (works with or without lane markings).

A strip of road just ahead of the car (the same 1 m grid area) is re-projected
into a fine top-down view (40 px/m) every frame.  In that view the road moves
straight towards the car, so frame-to-frame the asphalt texture (grain,
cracks, patches, tar seams, marking edges) is simply shifted by
``speed / fps`` metres.  The shift is found by normalised cross-correlation:
coarse search over the full speed range at 1/4 resolution, then a fine search
with sub-pixel (parabolic) peak interpolation.

Robustness:
* high-pass filtering keeps texture and removes lighting gradients;
* a slowly-updated background (mean of recent patches) is subtracted, which
  removes things that do *not* move with the road - windscreen reflections,
  the bonnet, dashcam text, the car's own shadow;
* a measurement is only accepted with a strong, unambiguous correlation peak,
  and not when a vehicle ahead intrudes into the strip.

The result is a displacement per frame; time to cross 1 m = 1 / speed.
"""
from __future__ import annotations

import cv2
import numpy as np

from .camera import CameraModel


class SurfaceOdometry:
    def __init__(self, cam: CameraModel, fps: float, ppm: float = 16.0, half_width_m: float = 6.0,
                 max_speed_mps: float = 55.0, src_scale: float = 1.0):
        self.cam = cam
        self.fps = fps
        self.ppm = ppm
        self.z0 = cam.z_min + 0.3                 # stay clear of the bonnet / frame bottom
        # Only use road close enough that the picture has real detail there: the image
        # has f*h/Z^2 rows per metre of road, so stop where the top-down view would need
        # more than 1.5x up-sampling (up-sampling creates a fixed pattern that does not
        # move with the road and would pull the estimate towards zero).
        f = cam.focal * src_scale
        z1 = float(np.sqrt(1.5 * f * cam.cam_height_m / ppm))
        z1 = max(z1, self.z0 + 2.5 + max_speed_mps / fps)   # keep enough strip for the search
        length_m = min(z1 - self.z0, 9.0)
        self.length = length_m
        self.far_s = (self.z0 - cam.z_min) + length_m   # strip end, in grid "s" metres
        W = int(round(2 * half_width_m * ppm))
        H = int(round(length_m * ppm))
        self.size = (W, H)
        # strip pixel (px, py) -> ground (X, Z) -> image
        z_far = self.z0 + length_m
        P2G = np.array([[1.0 / ppm, 0, -half_width_m], [0, -1.0 / ppm, z_far], [0, 0, 1.0]])
        self.P2I = cam.G2I @ P2G
        self.max_shift = int(np.ceil(max_speed_mps / fps * ppm)) + 4
        self.margin = 3                            # allows small negative shifts (jitter)
        self.lat = 3                               # lateral search (px) for steering / yaw
        self.bg = None
        self.n = 0
        self.quality: list[float] = []
        self.hist: list[np.ndarray] = []
        self.recent: list[float] = []

    def _strip(self, frame_small: np.ndarray) -> np.ndarray:
        g = cv2.warpPerspective(frame_small, self.P2I, self.size,
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        if g.ndim == 3:
            g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        g = g.astype(np.float32)
        g = g - cv2.GaussianBlur(g, (0, 0), 4.0)
        # remove everything that is the same all along the road (edge lines, worn wheel
        # tracks, kerbs, streaks): it matches at every shift and carries no speed information
        return g - g.mean(axis=0, keepdims=True)

    @staticmethod
    def _parabola(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-9 else 0.5 * (a - c) / d

    def _match(self, prev, cur, lo, hi, lat, scale):
        """Best shift (px at this scale) in [lo, hi] with NCC; returns (shift, peak, second)."""
        H, W = prev.shape
        a = self.margin_s(scale) + max(0, -lo)
        t_h = H - a - hi - 2
        if t_h < 12:
            return None
        tmpl = prev[a:a + t_h, lat:W - lat]
        search = cur[a + lo:a + hi + t_h, :]
        if tmpl.std() < 1e-3 or search.shape[0] < tmpl.shape[0]:
            return None
        res = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
        iy, ix = np.unravel_index(int(np.argmax(res)), res.shape)
        peak = float(res[iy, ix])
        sub = 0.0
        if 0 < iy < res.shape[0] - 1:
            sub = self._parabola(res[iy - 1, ix], res[iy, ix], res[iy + 1, ix])
        # second-best peak away from the best one (ambiguity check)
        r2 = res.copy()
        r2[max(iy - 3, 0):iy + 4, :] = -1
        second = float(r2.max()) if r2.size else -1.0
        return lo + iy + sub, peak, second

    def _profile(self, prev, cur, hi, lat):
        """NCC as a function of along-road shift (best over a small lateral search)."""
        H, W = prev.shape
        lo = -1
        a = 2
        t_h = H - a - hi - 2
        if t_h < 8:
            return None
        tmpl = prev[a:a + t_h, lat:W - lat]
        if tmpl.std() < 1e-3:
            return None
        res = cv2.matchTemplate(cur[a + lo:a + hi + t_h, :], tmpl, cv2.TM_CCOEFF_NORMED)
        return np.arange(lo, lo + res.shape[0]), res.max(axis=1)

    def margin_s(self, scale):
        return max(1, self.margin // scale)

    def _measure(self, p, c, max_shift):
        """Along-road shift (px) between two strips, or None if there is no clear moving peak."""
        k = 2
        ps = cv2.resize(p, None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA)
        cs = cv2.resize(c, None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA)
        prof = self._profile(ps, cs, max_shift // k + 1, max(1, self.lat // k))
        if prof is None:
            return None
        shifts, vals = prof
        # Static picture content (compression blocks, leftover reflections, resampling)
        # always correlates best at zero shift.  The road's own features show up as a
        # separate local peak at the true shift, so take the strongest *distinct* peak
        # away from zero; without one there is no measurement (NOT "zero speed").
        best = None
        for i in range(1, len(vals) - 1):
            if shifts[i] * k < 3:
                continue
            if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
                prom = vals[i] - max(vals[max(i - 4, 0):i].min(), vals[i + 1:i + 5].min())
                if vals[i] >= 0.3 and prom >= 0.06 and (best is None or vals[i] > vals[best]):
                    best = i
        if best is None:
            return None
        coarse = shifts[best] * k
        lo = max(int(np.floor(coarse)) - k, 1)
        hi = int(np.ceil(coarse)) + k
        r = self._match(p, c, lo, hi, self.lat, 1)
        if r is None or r[1] < 0.35:
            return None
        return r[0], r[1]

    def update(self, frame_small: np.ndarray, free_s: float | None) -> float:
        """Returns the measured displacement per frame (m), or NaN if it can't be measured."""
        cur = self._strip(frame_small)
        self.n += 1
        # background = things that stay put in the picture (reflections, bonnet, text);
        # used once enough frames are averaged that moving texture has blurred out
        if self.bg is None:
            self.bg = cur.copy()
        else:
            cv2.accumulateWeighted(cur, self.bg, 0.03)
        hist = self.hist
        hist.append(cur)
        if len(hist) > 9:
            hist.pop(0)
        if free_s is not None and free_s < self.far_s + 0.5:
            self.quality.append(0.0)            # a vehicle ahead is inside the strip
            return np.nan
        c = cur - self.bg if self.n > 30 else cur
        # Compare with the previous frame; at low speed the shift per frame is tiny, so
        # also try 4 and 8 frames back (keeps slow driving measurable).
        recent = [v for v in self.recent[-15:] if np.isfinite(v)]
        fast = len(recent) >= 3 and np.median(recent) > 0.15     # > ~0.15 m/frame
        for stride in ((1,) if fast else (1, 4, 8)):
            if len(hist) <= stride:
                break
            prev = hist[-1 - stride]
            p = prev - self.bg if self.n > 30 else prev
            r = self._measure(p, c, min(self.max_shift * stride, int(self.size[1] * 0.6)))
            if r is not None:
                self.quality.append(r[1])
                d = r[0] / self.ppm / stride
                self.recent.append(d)
                return d
        self.quality.append(0.0)
        self.recent.append(np.nan)
        return np.nan


class GroundFlow:
    """Sparse visual odometry on the ground plane (any road, any verge).

    Corner features are picked on the ground part of the frame (road, verges,
    cracks, shadows, bases of bushes/posts), tracked into the next frame with
    pyramidal Lucas-Kanade (forward-backward checked), and both positions are
    mapped onto the ground with the camera homography.  For points on the
    ground the along-road displacement equals the car's travel, so the robust
    mode of those displacements is the speed.

    It also tells whether the picture is actually moving at all (median image
    motion of all features), which keeps "stopped" from being reported on
    blurry or heavily compressed video while driving.
    """

    def __init__(self, cam: CameraModel, fps: float, max_z: float = 22.0):
        self.cam = cam
        self.fps = fps
        H, W = cam.height, cam.width
        # ground region of the frame: below the horizon, nearer than max_z, above the bonnet
        G2I = cam.G2I
        vz = (G2I @ np.array([0.0, max_z, 1.0]))
        v_far = int(vz[1] / vz[2])
        v_near = int(H * (1 - cam.bonnet_frac)) - 3
        # features are found and tracked at half resolution (4x cheaper, plenty of detail)
        self.sc = 0.5
        h2, w2 = int(H * self.sc), int(W * self.sc)
        self.mask = np.zeros((h2, w2), np.uint8)
        self.mask[max(int(v_far * self.sc), 0):max(int(v_near * self.sc), 0), :] = 255
        self.prev = None
        self.image_motion = np.nan      # median feature motion in pixels (last frame)
        self.n_tracked = 0

    def update(self, frame_small: np.ndarray, block_mask: np.ndarray | None = None) -> float:
        """Displacement per frame (m) from ground features, or NaN."""
        g = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (self.mask.shape[1], self.mask.shape[0]), interpolation=cv2.INTER_AREA)
        prev, self.prev = self.prev, g
        self.image_motion = np.nan
        self.n_tracked = 0
        if prev is None:
            return np.nan
        m = self.mask if block_mask is None else cv2.bitwise_and(self.mask, cv2.bitwise_not(block_mask))
        # real corners only (qualityLevel): points on a straight line or edge slide along
        # it and can't show along-road motion (aperture problem)
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=250, qualityLevel=0.03, minDistance=4,
                                     mask=m, blockSize=5)
        if p0 is None or len(p0) < 12:
            return np.nan
        lk = dict(winSize=(15, 15), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, **lk)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, p1, None, **lk)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1) < 0.5)
        if ok.sum() < 10:
            return np.nan
        a, b = p0.reshape(-1, 2)[ok] / self.sc, p1.reshape(-1, 2)[ok] / self.sc   # full-frame px
        mot = np.linalg.norm(b - a, axis=1)
        self.image_motion = float(np.median(mot))
        self.n_tracked = int(ok.sum())
        ga = self.cam.image_to_ground(a)
        gb = self.cam.image_to_ground(b)
        valid = (ga[:, 1] > 0.5) & (gb[:, 1] > 0.3) & (ga[:, 1] < 40)
        if valid.sum() < 10:
            return np.nan
        dz = (ga[valid, 1] - gb[valid, 1])           # travel towards the camera (m)
        moving = dz[np.abs(dz) > 0.03]               # ignore features that did not move
        if len(moving) < 8 or len(moving) < 0.2 * len(dz):
            return np.nan
        # robust mode: densest 10 % window of the sorted displacements, then its median
        srt = np.sort(moving)
        k = max(int(0.25 * len(srt)), 4)
        widths = srt[k:] - srt[:-k]
        i = int(np.argmin(widths))
        cluster = srt[i:i + k + 1]
        d = float(np.median(cluster))
        # the cluster must be tight relative to its value (consistent ground motion)
        if d <= 0 or (cluster[-1] - cluster[0]) > max(0.25 * d, 0.03):
            return np.nan
        return d
