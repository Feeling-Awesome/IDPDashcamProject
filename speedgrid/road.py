"""Road surface segmentation, boundary fitting and the turn-following metric grid.

All work is done in the metric bird's-eye view (BEV):
  * an adaptive Gaussian colour model of asphalt (CIE-Lab, seeded from the
    patch right in front of the car and updated online) gives a road mask;
  * morphology + connected components keep the carriageway the car is on;
  * row-wise extents are fitted with robust 2nd-order polynomials x(y) for the
    left / right road edges and temporally smoothed;
  * the centre line is parameterised by arc length, so grid "rungs" can be
    placed exactly 1 m apart *along the road* and perpendicular to it, which
    makes the grid bend with the turns.
"""
from __future__ import annotations

import cv2
import numpy as np

from .camera import CameraModel


class RoadModel:
    def __init__(self, cam: CameraModel, smooth: float = 0.25):
        self.cam = cam
        self.smooth = smooth
        self.mean = None          # Lab colour mean of asphalt
        self.cov = None
        self.params = None        # [a, b, c_left, c_right]  (shared-shape edge model)
        self.misses = 0
        self.mask = None
        self.ok = False

    # ------------------------------------------------------------ colour model
    F = 2   # road segmentation runs at 1/F of the BEV resolution (it only needs coarse detail)

    def _update_colour(self, lab: np.ndarray, sel: np.ndarray, rate: float):
        px = lab[sel]
        if len(px) > 6000:                      # a random-ish subsample is plenty
            px = px[:: len(px) // 6000]
        px = px.reshape(-1, 3).astype(np.float64)
        if len(px) < 50:
            return
        # Ignore bright outliers (lane paint) and very dark ones before fitting
        L = px[:, 0]
        lo, hi = np.percentile(L, [5, 90])
        px = px[(L >= lo) & (L <= hi)]
        if len(px) < 30:
            return
        m = px.mean(axis=0)
        c = np.cov(px.T) + np.diag([25.0, 4.0, 4.0])   # floor: tolerate shadows/texture
        if self.mean is None:
            self.mean, self.cov = m, c
        else:
            self.mean = (1 - rate) * self.mean + rate * m
            self.cov = (1 - rate) * self.cov + rate * c

    # --------------------------------------------------------------- per frame
    def update(self, bev_bgr: np.ndarray, marking_mask: np.ndarray | None = None):
        cam = self.cam
        F = self.F
        Wb, Hb = cam.bev_size
        w, h = Wb // F, Hb // F
        ppm = cam.ppm / F
        if getattr(self, "_valid_s", None) is None:
            self._valid_s = cv2.resize(cam.valid_bev_mask(), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        valid = self._valid_s
        small = cv2.resize(bev_bgr, (w, h), interpolation=cv2.INTER_AREA)
        # YCrCb: luma + two chroma axes (asphalt = neutral, grass/soil = coloured);
        # several times cheaper than Lab and separates the same things here
        lab = cv2.cvtColor(cv2.GaussianBlur(small, (3, 3), 0), cv2.COLOR_BGR2YCrCb)

        cx = w // 2
        seed = np.zeros((h, w), bool)
        seed[int(h - 4.0 * ppm):h - 1, int(cx - 1.0 * ppm):int(cx + 1.0 * ppm)] = True
        seed &= valid
        if self.mean is None:
            self._update_colour(lab, seed, 1.0)
        if self.mean is None:
            self.ok = False
            return

        # Mahalanobis distance to the asphalt model (L weighted down for shadows),
        # computed as ||A (x - mean)||^2 with A^T A = inverse covariance
        wgt = np.diag([0.35, 1.0, 1.0])
        icov = np.linalg.inv(wgt @ self.cov @ wgt)
        A = (np.linalg.cholesky(icov).T @ wgt).astype(np.float32)
        diff = cv2.subtract(lab, tuple(float(v) for v in self.mean) + (0.0,), dtype=cv2.CV_32F)
        t = cv2.transform(diff, A)
        d2 = cv2.transform(cv2.multiply(t, t), np.ones((1, 3), np.float32))
        road = ((d2 < 9.0) & valid).astype(np.uint8) * 255
        if marking_mask is not None:            # paint belongs to the road
            road |= cv2.resize(marking_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Fill paint / cracks across the road, then remove speckle
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (int(1.2 * ppm) | 1, int(0.6 * ppm) | 1))
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, k_close)
        road = cv2.morphologyEx(road, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # Keep only the component connected to the area in front of the car
        n, lbl = cv2.connectedComponents(road, connectivity=8)
        if n <= 1:
            self.misses += 1  # keep the previous edge model
            return
        ids = lbl[seed]
        ids = ids[ids > 0]
        if len(ids) == 0:
            self.misses += 1  # keep the previous edge model
            return
        road = (lbl == np.bincount(ids).argmax()).astype(np.uint8) * 255
        self.mask = cv2.resize(road, (Wb, Hb), interpolation=cv2.INTER_NEAREST)

        # Online colour adaptation from confidently-road pixels near the car
        # (lighting changes slowly, so every third frame is plenty)
        self._n = getattr(self, "_n", 0) + 1
        if self._n % 3 == 0:
            inner = cv2.erode(road, np.ones((5, 5), np.uint8)) > 0
            inner[: h // 2] = False
            self._update_colour(lab, inner, 0.12)

        self._fit_edges(road, valid)

    def _fit_edges(self, road: np.ndarray, valid: np.ndarray):
        """Joint parallel-edge fit:  x_left(u) = a u^2 + b u + cl,  x_right(u) = a u^2 + b u + cr
        with u = distance up the BEV (0 = nearest row, 1 = far end).

        Both edges share heading (b) and curvature (a), so an occluded or
        noisy side cannot bend independently.  Edge points are collected by
        following each edge outward from the car with a continuity check
        (a jump = a vehicle or a gap, so that side stops there) and gated
        against the previous frame's model.  The fit is regularised towards
        the previous model, checked for plausibility, then smoothed.

        ``road``/``valid`` are the low-resolution masks; the model is kept in
        full-resolution BEV pixels.
        """
        cam = self.cam
        F = self.F
        h, w = road.shape
        Hb = h * F
        ppm = cam.ppm
        prev = self.params
        u_max = min(28.0 / cam.bev_length_m, 0.8)

        rows = np.arange(h - 2, int(h * (1 - u_max)), -1)
        u = (Hb - rows * F) / Hb
        R = road[rows] > 0
        V = valid[rows]
        cols = np.arange(w)
        # column inside the road for each row: the car's own path, else the model centre
        c = np.full(len(rows), w // 2)
        if prev is not None:
            pc = np.clip(np.round(self._eval(prev, u, "c") / F).astype(int), 0, w - 1)
            use = ~R[np.arange(len(rows)), c]
            c[use] = pc[use]
        inside = R[np.arange(len(rows)), c]
        # run containing c: last non-road column left of c, first non-road right of c
        zl = np.where(~R, cols[None, :], -1)
        last_zero = np.maximum.accumulate(zl, axis=1)
        zr = np.where(~R, cols[None, :], w)
        first_zero = np.minimum.accumulate(zr[:, ::-1], axis=1)[:, ::-1]
        l = last_zero[np.arange(len(rows)), c] + 1
        r = first_zero[np.arange(len(rows)), c] - 1
        width_ok = (r - l) * F >= 2.5 * ppm
        # an edge touching the visible-area border is not a real road edge
        l_ok = (l > 1) & V[np.arange(len(rows)), np.clip(l - 2, 0, w - 1)]
        r_ok = (r < w - 2) & V[np.arange(len(rows)), np.clip(r + 2, 0, w - 1)]
        xl, xr = (l.astype(float) + 0.5) * F, (r.astype(float) + 0.5) * F

        jump, gate = 0.45 * ppm, 1.2 * ppm
        ok_l = inside & width_ok & l_ok
        ok_r = inside & width_ok & r_ok

        def collect(model):
            """Follow each edge outward from the car (vectorised): start at the first
            point inside the gate, stop at the first gate exit or sideways jump."""
            pts = {}
            for side, x, ok in (("l", xl, ok_l), ("r", xr, ok_r)):
                idx = np.flatnonzero(ok)
                if model is not None:
                    g = np.abs(x[idx] - self._eval(model, u[idx], side)) <= gate
                else:
                    g = np.ones(len(idx), bool)
                first = np.flatnonzero(g)
                if len(first) == 0:
                    pts[side] = (np.zeros(0), np.zeros(0))
                    continue
                st = first[0]
                xs = x[idx[st:]]
                bad = (~g[st + 1:]) | (np.abs(np.diff(xs)) > jump)
                stop = np.flatnonzero(bad)
                end = st + 1 + (stop[0] if len(stop) else len(xs) - 1)
                keep = idx[st:end]
                pts[side] = (u[keep], x[keep])
            return pts

        def support(th):
            """How many observed edge points (both sides, all rows) the model explains."""
            tol = 0.3 * ppm
            sl = np.abs(xl - self._eval(th, u, "l")) <= tol
            sr = np.abs(xr - self._eval(th, u, "r")) <= tol
            return int((sl & ok_l).sum() + (sr & ok_r).sum())

        # hypothesis 1: track the previous model (gated, regularised)
        tracked = None
        if prev is not None:
            p1 = collect(prev)
            tracked = self._joint_fit(p1["l"], p1["r"], prev)
            if tracked is not None and not self._plausible(tracked, prev):
                tracked = None
        # hypothesis 2: detect afresh (no gate, no prior) - lets the model recover
        # after an occlusion or a bad lock instead of staying stuck on it
        fresh = None
        self._k = getattr(self, "_k", 0) + 1
        if tracked is None or self._k % 4 == 0:
            p2 = collect(None)
            fresh = self._joint_fit(p2["l"], p2["r"], None)
            if fresh is not None and not self._plausible(fresh, None):
                fresh = None

        new, reset = tracked, False
        if fresh is not None:
            if tracked is None:
                if prev is None or self.misses >= 5 or support(fresh) > 1.5 * max(support(prev), 1):
                    new, reset = fresh, True
            elif support(fresh) > 1.3 * max(support(tracked), 10):
                new, reset = fresh, True

        if new is None:
            self.misses += 1
            if self.misses > 45:            # lost for ~1.5 s: allow re-initialisation
                self.params = None
                self.ok = False
            return
        self.misses = 0
        a = 0.6 if reset else self.smooth
        self.params = new if prev is None else (1 - a) * prev + a * new
        self.ok = True

    def _joint_fit(self, lp, rp, prev):
        ul, xl_ = lp
        ur, xr_ = rp
        n_l, n_r = len(ul), len(ur)
        if (n_l < 8 and n_r < 8) or (prev is None and (n_l < 8 or n_r < 8)):
            return None
        A = np.zeros((n_l + n_r, 4))
        A[:n_l, 0], A[:n_l, 1], A[:n_l, 2] = ul * ul, ul, 1.0
        A[n_l:, 0], A[n_l:, 1], A[n_l:, 3] = ur * ur, ur, 1.0
        rows = A
        rhs = np.concatenate([xl_, xr_])
        y = rhs
        ppm = self.cam.ppm
        # prior towards previous model (strong on shape, weak on position)
        if prev is not None:
            lam = np.sqrt(np.array([4.0, 2.0, 0.3, 0.3]) * max(len(y), 1) / 40.0)
            lam[2] = lam[2] if n_l >= 8 else 10.0    # hold an unseen side in place
            lam[3] = lam[3] if n_r >= 8 else 10.0
            Ap = np.diag(lam)
            yp = lam * prev
        else:
            Ap = np.diag([0.5, 0.05, 0.0, 0.0])
            yp = np.zeros(4)
        keep = np.ones(len(y), bool)
        theta = None
        for _ in range(4):
            M = np.vstack([A[keep], Ap])
            v = np.concatenate([y[keep], yp])
            theta, *_ = np.linalg.lstsq(M, v, rcond=None)
            res = np.abs(A @ theta - y)
            thr = max(3.0 * 1.4826 * np.median(res[keep]), 0.25 * ppm)
            nk = res < thr
            if nk.sum() < 8 or np.array_equal(nk, keep):
                break
            keep = nk
        return theta

    def _plausible(self, th, prev) -> bool:
        cam = self.cam
        ppm, L = cam.ppm, cam.bev_length_m
        a, b, cl, cr = th
        width = (cr - cl) / ppm
        if not (2.8 <= width <= 2 * cam.bev_half_width_m + 4):
            return False
        if abs(a) > ppm * L * L / (2 * 100.0):        # tighter than ~100 m radius
            return False
        if abs(b) > np.tan(np.radians(20)) * L * ppm:  # heading within +-20 deg
            return False
        if prev is not None:
            pw = (prev[3] - prev[2]) / ppm
            if abs(width - pw) > max(0.25 * pw, 1.5):
                return False
        return True

    @staticmethod
    def _eval(th, u, side):
        a, b, cl, cr = th
        c = {"l": cl, "r": cr, "c": (cl + cr) / 2}[side]
        return a * u * u + b * u + c

    # ----------------------------------------------------------- geometry API
    def band_mask(self, margin_m: float = 0.8) -> np.ndarray | None:
        """Pixels between the fitted road edges (+ margin), i.e. where paint may be."""
        if not self.ok:
            return None
        Wb, Hb = self.cam.bev_size
        ys = np.arange(Hb)
        xl, xr = self.edges_at(ys)
        m = margin_m * self.cam.ppm
        xs = np.arange(Wb)[None, :]
        band = (xs >= (xl - m)[:, None]) & (xs <= (xr + m)[:, None])
        return band.astype(np.uint8) * 255

    def edges_at(self, y: np.ndarray):
        Hb = self.cam.bev_size[1]
        u = (Hb - np.asarray(y, float)) / Hb
        return self._eval(self.params, u, "l"), self._eval(self.params, u, "r")


class RoadGrid:
    """Centre line with arc length + grid rungs 1 m apart along the road.

    ``s`` = distance along the road measured from the bottom of the BEV
    (the nearest visible ground point), in metres.
    """

    def __init__(self, cam: CameraModel, road: RoadModel | None, first_m: float, n_lines: int,
                 spacing_m: float = 1.0):
        self.cam = cam
        Wb, Hb = cam.bev_size
        ppm = cam.ppm
        ys = np.arange(Hb - 1, -1, -1, dtype=float)
        if road is not None and road.ok:
            xl, xr = road.edges_at(ys)
        else:
            xl = np.full_like(ys, Wb / 2 - 1.75 * ppm)
            xr = np.full_like(ys, Wb / 2 + 1.75 * ppm)
        self.xl, self.xr, self.ys = xl, xr, ys
        xc = (xl + xr) / 2.0
        seg = np.hypot(np.diff(xc), np.diff(ys))
        self.s = np.concatenate([[0.0], np.cumsum(seg)]) / ppm
        self.cx, self.cy = xc, ys
        self.half_w = (xr - xl) / 2.0
        tx = np.gradient(xc)
        ty = np.gradient(ys)
        n = np.hypot(tx, ty)
        self.tx, self.ty = tx / n, ty / n
        self.spacing = spacing_m
        self.line_s = first_m + spacing_m * np.arange(n_lines)
        self._centre_pts = np.column_stack([xc, ys]).astype(np.float32)

    # point in BEV -> (s along road, lateral offset in m)
    def to_road_coords(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2))
        ppm = self.cam.ppm
        d2 = ((pts[:, None, 0] - self._centre_pts[None, :, 0]) ** 2 +
              (pts[:, None, 1] - self._centre_pts[None, :, 1]) ** 2)
        j = np.argmin(d2, axis=1)
        dx, dy = pts[:, 0] - self.cx[j], pts[:, 1] - self.cy[j]
        # tangent (tx, ty) points towards decreasing y, i.e. increasing s
        along = dx * self.tx[j] + dy * self.ty[j]
        lat = -(dx * self.ty[j] - dy * self.tx[j])
        return np.column_stack([self.s[j] + along / ppm, lat / ppm])

    def free_distance(self, road_mask: np.ndarray | None, min_free_m: float = 3.0) -> float:
        """Distance (m) along the centre line until the lane ahead is blocked by a
        vehicle.  The grid is only drawn up to here.

        A vehicle in the bird's-eye view is not a small hole: everything above
        the ground smears away from the camera, so it blocks the lane for many
        metres.  Small gaps in the road mask (frame border, shadows, cracks,
        patches) are therefore ignored: we need >= 1.5 m blocked in a row *and*
        >= 70 % of the following 6 m blocked.
        """
        if road_mask is None:
            return float(self.s[-1])
        ppm = self.cam.ppm
        Hm, Wm = road_mask.shape
        half = int(1.0 * ppm)
        rows = np.clip(self.ys.astype(int), 0, Hm - 1)
        cols = np.clip(np.round(self.cx).astype(int)[:, None] + np.arange(-half, half + 1)[None, :], 0, Wm - 1)
        frac = (road_mask[rows[:, None], cols] > 0).mean(axis=1)
        blocked = frac < 0.5
        seen = np.flatnonzero(~blocked)
        if len(seen) == 0:
            return float(self.s[-1])
        start = seen[0]
        need, look = int(1.5 * ppm), int(6.0 * ppm)
        b = blocked.astype(np.int32)
        csum = np.concatenate([[0], np.cumsum(b)])
        n = len(b)
        i = np.arange(start, n)
        run_ok = (i + need <= n) & ((csum[np.minimum(i + need, n)] - csum[i]) == need)
        j = np.minimum(i + look, n)
        smear_ok = (csum[j] - csum[i]) >= 0.7 * (j - i)
        hit = np.flatnonzero(run_ok & smear_ok)
        if len(hit) == 0:
            return float(self.s[-1])
        return float(max(self.s[i[hit[0]]], min_free_m))

    def rung_bev(self, s_m: float, samples: int = 24) -> np.ndarray | None:
        """Grid rung at distance ``s_m``: perpendicular to the road centre line,
        ending exactly on the left / right edge curves (BEV pixels)."""
        if s_m <= 0 or s_m >= self.s[-1]:
            return None
        j = int(np.searchsorted(self.s, s_m))
        c = np.array([self.cx[j], self.cy[j]])
        normal = np.array([-self.ty[j], self.tx[j]])
        hw = max(self.half_w[j], 1.0)
        t = np.linspace(-2.5 * hw, 2.5 * hw, 801)
        p = c[None, :] + t[:, None] * normal[None, :]
        yi = np.clip(np.round(self.ys[0] - p[:, 1]).astype(int), 0, len(self.ys) - 1)
        fl = p[:, 0] - self.xl[yi]
        fr = p[:, 0] - self.xr[yi]
        mid = len(t) // 2
        li = np.flatnonzero(np.diff(np.sign(fl[: mid + 1])) != 0)
        ri = np.flatnonzero(np.diff(np.sign(fr[mid:])) != 0)
        tl = t[li[-1]] if len(li) else -hw
        tr = t[mid + ri[0]] if len(ri) else hw
        tt = np.linspace(tl, tr, samples)
        return c[None, :] + tt[:, None] * normal[None, :]

    def edge_bev(self, side: str, s_max: float) -> np.ndarray:
        m = self.s <= s_max
        x = self.xl if side == "left" else self.xr
        return np.column_stack([x[m], self.ys[m]])
