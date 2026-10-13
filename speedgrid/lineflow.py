"""Speed from the motion of the dashed-line *pattern* (robust at large per-frame motion).

Following individual dash ends breaks down when the car moves a large part of
the dash spacing between frames (fast driving, low frame rate, fast-forwarded
video): every dash looks like its neighbour.  Instead, the brightness profile
along each dashed lane line (dash, gap, dash, gap ... over ~25 m) is extracted
from the top-down view every frame and matched against the previous frame's
profile with normalised cross-correlation.  The pattern shift is the distance
travelled; time to cross 1 m of grid = frame time / shift.

Ambiguity: a regular dash pattern also matches one full period further on.
The smallest forward shift is taken at start-up, afterwards the candidate
closest to the recent speed (motion is continuous).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .camera import CameraModel


class LinePatternFlow:
    def __init__(self, cam: CameraModel, fps: float, max_speed_mps: float = 160.0, range_m: float = 26.0):
        self.cam = cam
        self.fps = fps
        self.ppm = cam.ppm
        self.L = int(range_m * cam.ppm)
        self.max_shift = int(np.ceil(max_speed_mps / fps * cam.ppm)) + 2
        self.offsets: deque = deque(maxlen=600)      # lateral positions of dash ends (m)
        self.prev: dict[float, np.ndarray] = {}
        self.recent: deque = deque(maxlen=20)
        self.moving = False

    def observe_dashes(self, lateral_m: list[float]):
        self.offsets.extend(lateral_m)

    def lines(self) -> list[float]:
        """Lateral positions (m, relative to the road centre line) of dashed lines."""
        if len(self.offsets) < 15:
            return []
        v = np.array(self.offsets)
        bins = np.round(v / 0.4) * 0.4
        vals, cnt = np.unique(bins, return_counts=True)
        out = []
        for b, c in sorted(zip(vals, cnt), key=lambda t: -t[1]):
            if c >= max(10, 0.15 * len(v)) and all(abs(b - o) > 1.5 for o in out):
                out.append(float(np.median(v[np.abs(v - b) <= 0.4])))
        return out[:3]

    def _is_dash_pattern(self, prof: np.ndarray) -> bool:
        """Does this profile look like a dashed lane line (>= 2 separate dashes of
        sensible length, mostly gaps)?  Rejects patches, vehicles, solid lines."""
        if len(prof) < 40 or prof.max() < 20:
            return False
        on = prof > max(18.0, 0.45 * float(np.percentile(prof, 99)))
        frac = on.mean()
        if not (0.05 <= frac <= 0.6):
            return False
        idx = np.flatnonzero(on)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 3) + 1)
        lens = [len(r) / self.ppm for r in runs]
        good = [l for l in lens if 0.4 <= l <= 7.0]
        return len(good) >= 2 and len(good) >= 0.6 * len(lens)

    def _profile(self, tophat: np.ndarray, grid, lat_m: float, free_s: float | None = None) -> np.ndarray:
        H, W = tophat.shape
        n = min(self.L, len(grid.cx))
        if free_s is not None:                     # never look at a vehicle ahead
            n = min(n, max(int((free_s - 1.0) * self.ppm), 0))
        idx = np.arange(n)                         # 0 = nearest row (grid arrays run near -> far)
        ys = grid.ys[idx].astype(int)
        xs = np.round(grid.cx[idx] + lat_m * self.ppm).astype(int)
        cols = np.clip(xs[:, None] + np.arange(-4, 5)[None, :], 0, W - 1)
        prof = tophat[np.clip(ys, 0, H - 1)[:, None], cols].max(axis=1).astype(np.float32)
        return prof

    def _shift(self, a: np.ndarray, b: np.ndarray, hint: float | None):
        """Shift (px) such that b(s) ~ a(s + d): the pattern came d closer."""
        n = len(a)
        cands = []
        res = np.full(self.max_shift + 1, -1.0)
        for d in range(2, min(self.max_shift, n - 30) + 1):
            x, y = a[d:], b[:n - d]
            sx, sy = x.std(), y.std()
            if sx < 1.0 or sy < 1.0:
                continue
            res[d] = float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))
        for d in range(3, len(res) - 1):
            if res[d] >= 0.6 and res[d] >= res[d - 1] and res[d] >= res[d + 1]:
                sub = 0.0
                den = res[d - 1] - 2 * res[d] + res[d + 1]
                if abs(den) > 1e-9:
                    sub = 0.5 * (res[d - 1] - res[d + 1]) / den
                cands.append((d + sub, res[d]))
        if not cands:
            return None
        best_val = max(c[1] for c in cands)
        good = [c for c in cands if c[1] >= 0.8 * best_val]
        if hint is None:
            return min(good, key=lambda c: c[0])[0]           # smallest forward shift
        near = [c for c in good if abs(c[0] - hint) <= 0.4 * hint + 2]
        if not near:
            return None
        return min(near, key=lambda c: abs(c[0] - hint))[0]

    def update(self, tophat: np.ndarray | None, grid, free_s: float | None = None) -> float:
        """Displacement per frame (m) from the dashed-line pattern, or NaN."""
        if tophat is None:
            self.prev = {}
            return np.nan
        lines = self.lines()
        cur = {}
        for l in lines:
            prof = self._profile(tophat, grid, l, free_s)
            if self._is_dash_pattern(prof):
                cur[round(l, 1)] = prof
        rec = [r for r in self.recent if np.isfinite(r)]
        hint = float(np.median(rec)) * self.ppm if len(rec) >= 3 else None
        est = []
        for key, prof in cur.items():
            prev = self.prev.get(key)
            if prev is None:
                continue
            m = min(len(prev), len(prof))
            if m < 60:
                continue
            d = self._shift(prev[:m], prof[:m], hint)
            if d is not None:
                est.append(d / self.ppm)
        self.prev = cur
        v = float(np.median(est)) if est else np.nan
        self.recent.append(v)
        self.moving = sum(1 for r in list(self.recent)[-5:] if np.isfinite(r) and r * self.fps > 2.0) >= 3
        if len(self.recent) == self.recent.maxlen and not any(np.isfinite(self.recent)):
            self.recent.clear()
        return v
