"""Tracking of lane-marking endpoints and grid-line crossing timing.

Each dash endpoint is tracked in road coordinates (s = metres along the road,
d = lateral offset) with a constant-velocity Kalman filter.  Data association
is a gated nearest-neighbour assignment (cheapest pairs first).

Whenever a track's filtered position passes a grid line, the crossing time is
interpolated *between frames* (linear interpolation of s(t)), which gives
sub-frame timing even when the marking moves almost a metre per frame at
highway speed.  When the same marking crosses two consecutive grid lines
(1 m apart) the speed sample is  v = 1 m / (t_k - t_{k+1}).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    tid: int
    kind: str
    x: np.ndarray            # [s, v]  (v = ds/dt, negative when approaching)
    P: np.ndarray
    d: float                 # lateral offset (m)
    t_last: float
    hits: int = 1
    misses: int = 0
    s_meas_prev: tuple[float, float] | None = None   # (t, s) of previous update
    crossings: dict[int, float] = field(default_factory=dict)
    bev_xy: tuple[float, float] = (0.0, 0.0)


@dataclass
class SpeedSample:
    t: float
    speed_mps: float
    track: int
    line_pair: tuple[int, int]
    dt: float


class MarkingTracker:
    def __init__(self, line_s: np.ndarray, max_speed_mps: float = 70.0):
        self.line_s = np.asarray(line_s, float)
        self.tracks: list[Track] = []
        self.next_id = 1
        self.max_speed = max_speed_mps
        self.samples: list[SpeedSample] = []
        self.recent_crossings: list[tuple[float, int]] = []  # (t, line index) for overlay
        self.global_v: float | None = None    # ds/dt estimate (negative)
        self.stationary = False               # set by the ego-motion detector each frame
        self.max_pair_dt = 1.0 / (6.0 / 3.6)  # slowest measurable: 6 km/h over 1 m

    # ----------------------------------------------------------- Kalman bits
    @staticmethod
    def _predict(tr: Track, t: float):
        dt = t - tr.t_last
        F = np.array([[1, dt], [0, 1.0]])
        q = 4.0  # m/s^2 accel noise (car speed changes slowly)
        Q = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        return F @ tr.x, F @ tr.P @ F.T + Q

    def step(self, t: float, dets: list[tuple[float, float, str, float, float]]):
        """dets: (s, d, kind, bev_x, bev_y)."""
        new_samples = []
        preds = [self._predict(tr, t) for tr in self.tracks]
        nt, nd = len(self.tracks), len(dets)
        assigned_t, assigned_d = set(), set()
        if nt and nd:
            C = np.full((nt, nd), 1e6)
            for i, tr in enumerate(self.tracks):
                xp, Pp = preds[i]
                v_known = tr.hits >= 2 or self.global_v is not None
                for j, (s, d, kind, _, _) in enumerate(dets):
                    if kind != tr.kind or abs(d - tr.d) > 0.6:
                        continue
                    ds = s - xp[0]
                    if v_known:
                        gate = 0.8 + 3.0 * np.sqrt(Pp[0, 0])
                        if abs(ds) > gate:
                            continue
                    else:
                        # velocity unknown: marking must move towards the car,
                        # pick the smallest plausible displacement
                        max_disp = self.max_speed * (t - tr.t_last) + 0.3
                        if not (-max_disp <= ds <= 0.3):
                            continue
                    C[i, j] = abs(ds) + 1.5 * abs(d - tr.d)
            for i, j in _assign(C):
                assigned_t.add(i)
                assigned_d.add(j)
                new_samples += self._update(self.tracks[i], preds[i], t, dets[j])

        # unmatched tracks
        keep = []
        for i, tr in enumerate(self.tracks):
            if i not in assigned_t:
                tr.misses += 1
            if tr.misses <= 3 and self.line_s[0] - 2.0 < tr.x[0] < self.line_s[-1] + 25:
                keep.append(tr)
        self.tracks = keep

        # new tracks
        for j, (s, d, kind, bx, by) in enumerate(dets):
            if j in assigned_d:
                continue
            v0 = self.global_v if self.global_v is not None else 0.0
            pv = 4.0 if self.global_v is not None else 400.0
            tr = Track(self.next_id, kind, np.array([s, v0]), np.diag([0.05, pv]), d, t,
                       s_meas_prev=(t, s), bev_xy=(bx, by))
            self.next_id += 1
            self.tracks.append(tr)

        # global velocity from mature tracks (helps initialise new ones)
        vs = [tr.x[1] for tr in self.tracks if tr.hits >= 4 and tr.misses == 0]
        if len(vs) >= 1:
            self.global_v = float(np.median(vs))
        self.samples += new_samples
        return new_samples

    def _update(self, tr: Track, pred, t: float, det):
        s, d, kind, bx, by = det
        xp, Pp = pred
        R = (0.04 + 0.006 * max(s, 0)) ** 2
        if tr.hits == 1 and self.global_v is None:
            # second observation: initialise velocity from finite difference
            dt = t - tr.t_last
            v = (s - tr.x[0]) / dt if dt > 0 else 0.0
            x_new = np.array([s, v])
            P_new = np.diag([R, 2 * R / max(dt, 1e-3) ** 2 + 1.0])
        else:
            H = np.array([[1.0, 0.0]])
            S = H @ Pp @ H.T + R
            K = Pp @ H.T / S
            x_new = xp + (K * (s - xp[0])).ravel()
            P_new = (np.eye(2) - K @ H) @ Pp
        s_prev_t, s_prev = tr.t_last, tr.x[0]
        tr.x, tr.P, tr.t_last = x_new, P_new, t
        tr.d = 0.8 * tr.d + 0.2 * d
        tr.hits += 1
        tr.misses = 0
        tr.bev_xy = (bx, by)
        return self._crossings(tr, s_prev_t, s_prev, t, x_new[0])

    def _crossings(self, tr: Track, t0: float, s0: float, t1: float, s1: float):
        out = []
        v_k = -tr.x[1]                       # Kalman speed of this marking (m/s, >0 = approaching)
        # forget crossings that are too old to pair, or when the marking isn't moving:
        # a marking jittering over a line while the car stands still must never pair up
        tr.crossings = {k: tc for k, tc in tr.crossings.items() if t1 - tc <= self.max_pair_dt}
        if self.stationary or v_k < 1.0:
            tr.crossings.clear()
            return out
        if s1 >= s0:
            return out
        idx = np.flatnonzero((self.line_s < s0) & (self.line_s >= s1))
        for k in sorted(idx, reverse=True):
            g = self.line_s[k]
            tc = t0 + (s0 - g) / (s0 - s1) * (t1 - t0)
            tr.crossings[int(k)] = tc
            self.recent_crossings.append((tc, int(k)))
            prev = tr.crossings.get(int(k) + 1)
            if prev is None or tr.hits < 3:
                continue
            dt = tc - prev
            if not (0 < dt <= self.max_pair_dt):
                continue
            v = (self.line_s[k + 1] - g) / dt
            # must agree with the marking's own filtered motion (rejects mis-associations)
            if 0.5 < v < self.max_speed and abs(v - v_k) <= 0.35 * v_k:
                out.append(SpeedSample(tc, v, tr.tid, (int(k) + 1, int(k)), dt))
        return out


def _assign(C: np.ndarray):
    """Gated greedy assignment: repeatedly take the cheapest remaining
    (track, detection) pair.  With tight gates this matches the optimal
    (Hungarian) assignment in practice and needs no SciPy."""
    pairs = []
    if C.size == 0:
        return pairs
    order = np.argsort(C, axis=None)
    used_r, used_c = set(), set()
    for flat in order:
        i, j = divmod(int(flat), C.shape[1])
        if C[i, j] >= 1e6:
            break
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        pairs.append((i, j))
    return pairs


METHOD_NAMES = {"M": "lane markings", "S": "road surface"}


def window_row(w: int, a: float, b: float, seg: np.ndarray, st: np.ndarray, n_samp: int,
               methods=None) -> dict:
    row = {"window": w + 1, "start_s": round(a, 2), "end_s": round(b, 2), "samples": n_samp,
           "speed_mps": None, "speed_kmh": None, "status": "no data", "method": ""}
    known = seg[~np.isnan(seg)]
    if methods is not None:
        ms = [m for m in methods if m in METHOD_NAMES]
        if ms:
            nm, ns = ms.count("M"), ms.count("S")
            row["method"] = (METHOD_NAMES["M"] if ns == 0 else METHOD_NAMES["S"] if nm == 0
                             else "markings + surface")
    if len(seg) and st.mean() >= 0.9:
        row.update(speed_mps=0.0, speed_kmh=0.0, status="stopped", method="")
    elif len(known) >= 0.5 * len(seg) and len(known):
        v = float(np.mean(known))          # time-average over the window
        status = "moving" if st.mean() < 0.1 else "stopping/starting"
        if status == "moving" and n_samp < 4 and row["method"] == METHOD_NAMES["M"]:
            status = "low data"
        row.update(speed_mps=round(v, 2), speed_kmh=round(v * 3.6, 1), status=status)
    return row


def window_bounds(w: int, window: float, duration: float, fps: float):
    a, b = w * window, min((w + 1) * window, duration)
    return a, b, int(round(a * fps)), int(round(b * fps))


def aggregate_windows(samples: list[SpeedSample], timeline: np.ndarray, stationary: np.ndarray,
                      fps: float, window: float = 3.0, methods=None):
    rows = []
    duration = len(timeline) / fps
    n_win = int(np.ceil(duration / window - 1e-9))
    ts = sorted(x.t for x in samples)
    for w in range(max(n_win, 1)):
        a, b, fa, fb = window_bounds(w, window, duration, fps)
        n_samp = bisect.bisect_left(ts, b) - bisect.bisect_left(ts, a)
        rows.append(window_row(w, a, b, timeline[fa:fb], stationary[fa:fb], n_samp,
                               None if methods is None else methods[fa:fb]))
    return rows


class LiveTimeline:
    """Per-frame speed, built incrementally while the video is analysed.

    Frame k's speed is (in order of preference):
      0 when the car is stopped;
      "M": robust median of the lane-marking 1 m crossing samples within +-half s;
      "S": robust median of the road-surface (texture) measurements within +-half s;
      bridged linearly across short gaps; otherwise unknown.

    Frame k's value is final once the samples within +-half of it have all
    arrived and any gap containing it can be bridged (needs ``bridge`` s of
    look-ahead).  This lets the video be drawn in the same single pass as the
    analysis with only a few seconds of frame buffering.
    """

    def __init__(self, fps: float, half: float = 0.75, bridge: float = 3.0):
        self.fps, self.half, self.bridge = fps, half, bridge
        self.ts: list[float] = []
        self.vs: list[float] = []
        self.raw: list[float] = []
        self.final: list[float] = []
        self.surface: list[float] = []      # per-frame road-surface speed (m/s) or NaN
        self.method: list[str] = []         # per-frame source: Z(stopped) M S I(bridged) 
        self.last_known: int | None = None
        self.bridge_frames = bridge * fps

    def add(self, samples: list[SpeedSample]):
        for x in samples:
            i = bisect.bisect_right(self.ts, x.t)
            self.ts.insert(i, x.t)
            self.vs.insert(i, x.speed_mps)

    @staticmethod
    def _robust(v: np.ndarray) -> float:
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        return float(np.median(v[np.abs(v - med) <= max(3 * mad, 0.05 * abs(med))]))

    def _raw_value(self, k: int):
        t = k / self.fps
        a = bisect.bisect_left(self.ts, t - self.half)
        b = bisect.bisect_left(self.ts, t + self.half)
        n_mark = b - a
        if n_mark >= 4:                     # real dashed markings: the primary method
            return self._robust(np.array(self.vs[a:b])), "M"
        h = int(round(self.half * self.fps))
        seg = np.array(self.surface[max(k - h, 0):k + h + 1], float)
        ok = seg[np.isfinite(seg)]
        if len(ok) >= max(3, 0.35 * len(seg)):
            return self._robust(ok), "S"
        if n_mark >= 2:                     # a few marking crossings, no surface reading
            return self._robust(np.array(self.vs[a:b])), "M"
        return np.nan, ""

    def compute_raw(self, upto: int, stationary: list[bool]):
        for k in range(len(self.raw), upto + 1):
            if stationary[k]:
                self.raw.append(0.0)
                self.method.append("Z")
            else:
                v, m = self._raw_value(k)
                self.raw.append(v)
                self.method.append(m)

    def finalize(self, upto: int, complete: bool = False) -> int:
        """Finalise frames while possible (all of them when ``complete``)."""
        n_raw = len(self.raw)
        while len(self.final) <= upto:
            j = len(self.final)
            r = self.raw[j]
            if not np.isnan(r):
                self.final.append(r)
                self.last_known = j
                continue
            # gap: interpolate between the neighbouring known frames if they are
            # close enough; decide as soon as the answer can no longer change
            prv = self.last_known
            if prv is None:
                self.final.append(np.nan)
                continue
            limit = prv + self.bridge_frames
            nxt = None
            for k in range(j + 1, min(n_raw, int(limit) + 1)):
                if not np.isnan(self.raw[k]):
                    nxt = k
                    break
            if nxt is None and not complete and n_raw - 1 < limit:
                break                       # not decidable yet: wait for more frames
            if nxt is not None:
                self.final.append(self.raw[prv] + (self.raw[nxt] - self.raw[prv]) * (j - prv) / (nxt - prv))
                self.method[j] = "I"
            else:
                self.final.append(np.nan)
        return len(self.final)
