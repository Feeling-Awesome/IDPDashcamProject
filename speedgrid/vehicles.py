"""Other vehicles: detection, tracking and absolute speed.

* Detection: YOLOX-S (COCO, Apache-2.0) run through OpenCV's DNN module on the
  band of the frame where the road is (horizon to bonnet), a few times per
  second.  Only cars, trucks, buses and motorbikes are kept.
* Tracking: boxes are linked over time by overlap with each track's predicted
  box (greedy, best overlap first).
* Distance: the bottom-centre of a box is where the vehicle touches the road.
  Mapping it through the same flat-road camera model as the 1 m grid gives its
  distance along the road in grid metres.  Far away that is coarse (one pixel
  is over a metre at 30 m), so it is fused with a distance from the box width
  (width x distance is constant for one vehicle), each weighted by its own
  expected error.
* Speed: a weighted straight-line fit of distance against time over about
  +-1 s gives how fast the vehicle moves across the grid lines relative to us;
  adding our own (ego) speed makes it absolute.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .camera import CameraModel

MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolox_s.onnx"
VEHICLE_CLASSES = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
_STRIDES = (8, 16, 32)


class VehicleDetector:
    """YOLOX-S on a horizontal band of the frame (much cheaper than the full square input)."""

    def __init__(self, model_path: str | os.PathLike = MODEL_PATH, input_w: int = 640, conf: float = 0.4):
        self.net = cv2.dnn.readNet(str(model_path))
        self.input_w = input_w
        self.conf = conf
        self._grids: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _grid(self, h, w):
        key = (h, w)
        if key not in self._grids:
            g, st = [], []
            for s in _STRIDES:
                yv, xv = np.mgrid[:h // s, :w // s]
                g.append(np.stack([xv, yv], -1).reshape(-1, 2))
                st.append(np.full((g[-1].shape[0], 1), s))
            self._grids[key] = (np.concatenate(g).astype(np.float32), np.concatenate(st).astype(np.float32))
        return self._grids[key]

    def detect(self, img: np.ndarray) -> list[tuple[np.ndarray, float, str]]:
        """Boxes (x1, y1, x2, y2) in ``img`` pixels, confidence, class name.  The image is
        scaled to the detector width and padded to a multiple of 32 rows."""
        H, W = img.shape[:2]
        if H < 16 or W < 16:
            return []
        r = self.input_w / W
        bh = int(round(H * r))
        ih = int(np.ceil(bh / 32) * 32)
        inp = np.full((ih, self.input_w, 3), 114, np.uint8)
        inp[:bh] = cv2.resize(img, (self.input_w, bh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        self.net.setInput(cv2.dnn.blobFromImage(inp))          # BGR, 0-255, no scaling (YOLOX)
        out = self.net.forward()[0]
        g, st = self._grid(ih, self.input_w)
        cls = out[:, 5:] * out[:, 4:5]
        cid = cls.argmax(1)
        conf = cls[np.arange(len(cls)), cid]
        keep = (conf >= self.conf) & np.isin(cid, list(VEHICLE_CLASSES))
        if not keep.any():
            return []
        o, g, st, conf, cid = out[keep], g[keep], st[keep], conf[keep], cid[keep]
        xy = (o[:, :2] + g) * st
        wh = np.exp(np.clip(o[:, 2:4], -10, 10)) * st
        boxes = np.concatenate([xy - wh / 2, wh], 1) / r
        idx = np.array(cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), self.conf, 0.5)).reshape(-1)
        res = []
        for i in idx:
            x, y, w, h = boxes[i]
            res.append((np.array([x, y, x + w, y + h], np.float32), float(conf[i]),
                        VEHICLE_CLASSES[int(cid[i])]))
        return res


@dataclass
class Obs:
    fi: int                  # frame index (all frames, including repeated ones)
    t: float                 # true time (s)
    box: np.ndarray          # x1, y1, x2, y2
    z: float                 # distance ahead from the box bottom (m), NaN if unusable
    x: float                 # lateral offset (m)
    conf: float


@dataclass
class VehicleTrack:
    tid: int
    kind: str
    obs: list[Obs] = field(default_factory=list)
    misses: int = 0

    def predicted_box(self, t: float) -> np.ndarray:
        b = self.obs[-1].box
        if len(self.obs) < 2:
            return b
        a = self.obs[-2]
        dt = self.obs[-1].t - a.t
        if dt <= 0:
            return b
        return b + (b - a.box) * (t - self.obs[-1].t) / dt


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class VehicleSpeeds:
    """Tracks vehicles over the video and answers 'which boxes and speeds on frame j'."""

    def __init__(self, cam: CameraModel, fps: float, speedup: float = 1.0, max_dist_m: float = 80.0,
                 half_window_s: float = 1.0, px_noise: float = 1.5, max_sd_kmh: float = 6.0):
        self.cam = cam
        self.fps = fps
        self.sp = max(float(speedup or 1.0), 0.01)
        self.max_dist = max_dist_m
        self.half = half_window_s
        self.px_noise = px_noise
        self.max_sd_kmh = max_sd_kmh
        self.tracks: list[VehicleTrack] = []
        self.done: list[VehicleTrack] = []
        self.next_id = 1
        H = cam.height
        self.horizon_y = float((cam.G2I @ np.array([0.0, 1e6, 1.0]))[1] / (cam.G2I @ np.array([0.0, 1e6, 1.0]))[2])
        self.bottom_y = H * (1.0 - cam.bonnet_frac) - 3

    # ------------------------------------------------------------- detection side
    def band(self) -> tuple[int, int]:
        """Rows to run the detector on: from well above the horizon (tall trucks) to the bonnet."""
        H = self.cam.height
        top = self.horizon_y - 0.22 * H
        return int(max(top, 0)), int(min(self.bottom_y + 3, H))

    def zoom_rect(self) -> tuple[int, int, int, int]:
        """Small region round the vanishing point where far vehicles are (x0, y0, x1, y1),
        in processing-frame pixels.  It is cut from the full-resolution frame, so far
        vehicles get several times more pixels than in the whole-band pass."""
        H, W = self.cam.height, self.cam.width
        vx = float(np.clip(self.cam.vp[0] if self.cam.vp else W / 2, 0.3 * W, 0.7 * W))
        x0, x1 = vx - 0.22 * W, vx + 0.22 * W
        y0, y1 = self.horizon_y - 0.07 * H, self.horizon_y + 0.11 * H
        return int(max(x0, 0)), int(max(y0, 0)), int(min(x1, W)), int(min(y1, self.bottom_y))

    @staticmethod
    def merge(full, zoom, zrect):
        """Combine both passes: a vehicle seen whole in the zoomed pass uses that (sharper) box."""
        x0, y0, x1, y1 = zrect
        keep_zoom = [d for d in zoom if d[0][0] > x0 + 2 and d[0][2] < x1 - 2 and d[0][1] > y0 + 2
                     and d[0][3] < y1 - 2]
        out = list(keep_zoom)
        for d in full:
            if all(_iou(d[0], z[0]) < 0.3 for z in keep_zoom):
                out.append(d)
        return out

    def observe(self, fi: int, t: float, dets):
        W = self.cam.width
        obs = []
        for box, conf, kind in dets:
            x1, y1, x2, y2 = box
            if (x2 - x1) < 10 or y2 <= self.horizon_y + 2:
                continue
            z = x = np.nan
            # the bottom edge is the road contact point unless the bonnet / frame bottom cuts it
            if y2 < self.bottom_y - 2:
                X, Z = self.cam.image_to_ground(np.array([[(x1 + x2) / 2, y2]]))[0]
                if 0.5 < Z < 1.5 * self.max_dist:
                    z, x = float(Z), float(X)
            obs.append((Obs(fi, t, box, z, x, conf), kind))

        live = [tr for tr in self.tracks]
        pairs = []
        for i, tr in enumerate(live):
            pb = tr.predicted_box(t)
            for j, (o, kind) in enumerate(obs):
                v = _iou(pb, o.box)
                if v >= 0.2:
                    pairs.append((v, i, j))
        used_t, used_o = set(), set()
        for v, i, j in sorted(pairs, reverse=True):
            if i in used_t or j in used_o:
                continue
            used_t.add(i)
            used_o.add(j)
            live[i].obs.append(obs[j][0])
            live[i].misses = 0
        keep = []
        for i, tr in enumerate(live):
            if i not in used_t:
                tr.misses += 1
            if tr.misses > 3:
                self.done.append(tr)
            else:
                keep.append(tr)
        self.tracks = keep
        for j, (o, kind) in enumerate(obs):
            if j not in used_o and o.conf >= 0.5:
                self.tracks.append(VehicleTrack(self.next_id, kind, [o]))
                self.next_id += 1

    def all_tracks(self):
        return self.done + self.tracks

    # ------------------------------------------------------------------ speed side
    @staticmethod
    def _wfit(ts, y, sd, t0):
        """Weighted straight-line fit with one outlier pass: (slope, intercept at t0, sd of slope)."""
        for _ in range(2):
            if len(ts) < 3 or ts[-1] - ts[0] < 0.5:
                return None
            w = 1.0 / sd ** 2
            A = np.stack([ts - t0, np.ones_like(ts)], 1)
            N = A.T @ (A * w[:, None])
            try:
                cov = np.linalg.inv(N)
            except np.linalg.LinAlgError:
                return None
            slope, icpt = cov @ (A.T @ (w * y))
            r = np.abs(y - (slope * (ts - t0) + icpt)) / sd
            keep = r <= max(3.0, 2.5 * float(np.median(r)))
            if keep.all():
                # inflate the error if the points scatter more than expected
                chi = float(np.sqrt(np.sum(r ** 2) / max(len(ts) - 2, 1)))
                return float(slope), float(icpt), float(np.sqrt(cov[0, 0]) * max(chi, 1.0))
            ts, y, sd = ts[keep], y[keep], sd[keep]
        return None

    def relative_speed(self, tr: VehicleTrack, t: float, with_sd: bool = False):
        """Speed relative to us at time t (see _relative).  Far vehicles get a longer
        window: their distance is coarser, and their speed changes slowly."""
        r = self._relative(tr, t, self.half)
        if not (np.isfinite(r[2]) and r[2] * 3.6 / self.sp <= self.max_sd_kmh) and \
                not (np.isfinite(r[1]) and r[1] > self.max_dist):
            r2 = self._relative(tr, t, 1.5 * self.half)
            if np.isfinite(r2[2]) and (not np.isfinite(r[2]) or r2[2] < r[2]):
                r = r2
        return r if with_sd else r[:2]

    def _relative(self, tr: VehicleTrack, t: float, half: float):
        """(relative speed in m/s of on-screen time, distance in m) at time t, NaN if unknown.
        Positive = moving away from us (faster than us).

        Two independent estimates over the same +-1 s window, combined by their errors:
          * the road-contact distance (bottom of the box, grid metres) against time;
          * the rate of growth of the box width: d(ln w)/dt = -(dZ/dt) / Z.
        Width is only used for boxes that are not cut by the frame edge.
        """
        nan = (np.nan, np.nan, np.nan)
        obs = [o for o in tr.obs if abs(o.t - t) <= half]
        if len(obs) < 4:
            return nan
        f, h, s = self.cam.focal, self.cam.cam_height_m, self.px_noise
        W = self.cam.width
        ts = np.array([o.t for o in obs])
        zb = np.array([o.z for o in obs])
        ok = np.isfinite(zb)
        fb = self._wfit(ts[ok], zb[ok], zb[ok] ** 2 / (f * h) * s + 0.05, t) if ok.sum() >= 3 else None
        if fb is None:
            return nan
        v_b, z0, sd_b = fb
        if not (0.5 < z0 < self.max_dist):
            return np.nan, z0, np.nan
        whole = np.array([o.box[0] > 3 and o.box[2] < W - 4 and o.box[3] < self.bottom_y - 2 for o in obs])
        w = np.array([o.box[2] - o.box[0] for o in obs], float)
        est, wgt = [v_b], [1 / sd_b ** 2]
        if whole.sum() >= 4:
            fw = self._wfit(ts[whole], np.log(w[whole]), s / w[whole], t)
            if fw is not None:
                g, _, sd_g = fw
                # the box shape changes slowly with viewing angle: floor the error at 2 %/s
                est.append(-z0 * g)
                wgt.append(1 / (z0 * max(sd_g, 0.02)) ** 2)
        v = float(np.dot(est, wgt) / np.sum(wgt))
        return v, z0, float(1 / np.sqrt(np.sum(wgt)))

    def boxes_at(self, j: int, t: float, ego_mps: float, hold_frames: int):
        """Vehicles to draw on frame j: (box, absolute km/h or NaN, distance m, track id)."""
        out = []
        for tr in self.all_tracks():
            if len(tr.obs) < 2 or j < tr.obs[0].fi or j > tr.obs[-1].fi + hold_frames:
                continue
            fis = [o.fi for o in tr.obs]
            k = int(np.searchsorted(fis, j))
            if k < len(fis) and fis[k] == j:
                box = tr.obs[k].box
            elif 0 < k < len(fis):
                a, b = tr.obs[k - 1], tr.obs[k]
                u = (j - a.fi) / max(b.fi - a.fi, 1)
                box = a.box + (b.box - a.box) * u
            else:
                box = tr.obs[-1].box
            rel, dist, sd = self.relative_speed(tr, t, with_sd=True)
            if np.isfinite(dist) and dist > self.max_dist:
                continue
            if not (np.isfinite(sd) and sd / self.sp * 3.6 <= self.max_sd_kmh):
                rel = np.nan                       # too uncertain to print: box only
            v = ego_mps + rel / self.sp if (np.isfinite(rel) and np.isfinite(ego_mps)) else np.nan
            out.append((box, v * 3.6 if np.isfinite(v) else np.nan, dist, tr.tid))
        return out

    def summary(self, ego_at) -> list[dict]:
        """Per-vehicle summary for results.json (ego_at(t) -> ego speed m/s)."""
        res = []
        for tr in self.all_tracks():
            if len(tr.obs) < 4:
                continue
            ts = [o.t for o in tr.obs]
            sp, samples = [], []
            for o in tr.obs:
                rel, dist, sd = self.relative_speed(tr, o.t, with_sd=True)
                e = ego_at(o.t)
                if (np.isfinite(rel) and np.isfinite(e) and dist <= self.max_dist
                        and sd / self.sp * 3.6 <= self.max_sd_kmh):
                    sp.append((e + rel / self.sp) * 3.6)
                    samples.append({"t": round(o.t, 3), "speed_kmh": round(sp[-1], 1),
                                    "distance_m": round(dist, 1),
                                    "box": [int(round(c)) for c in o.box]})
            res.append({"id": tr.tid, "type": tr.kind, "first_s": round(ts[0], 2), "last_s": round(ts[-1], 2),
                        "speed_kmh_median": round(float(np.median(sp)), 1) if sp else None,
                        "speed_kmh_min": round(float(np.min(sp)), 1) if sp else None,
                        "speed_kmh_max": round(float(np.max(sp)), 1) if sp else None,
                        "samples": samples})
        return res
