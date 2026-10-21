"""End-to-end video processing: calibration pre-pass, per-frame analysis,
annotated output video and the 3-second speed table."""
from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import subprocess
import sys
import threading
import time as _time
import queue
import bisect
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .camera import CameraModel, estimate_vanishing_point
from .markings import detect_markings, measure_lane_spacing
from .road import RoadGrid, RoadModel
from .egomotion import EgoMotion
from .odometry import GroundFlow, SurfaceOdometry
from .lineflow import LinePatternFlow
from .vehicles import MODEL_PATH, VehicleDetector, VehicleSpeeds
from .box3d import Vehicle3DModel
from .tracker import (LiveTimeline, MarkingTracker, SpeedSample, aggregate_windows, window_bounds,
                      window_row)

GRID_COLOUR = (0, 230, 255)      # yellow (BGR)
EDGE_COLOUR = (0, 230, 255)
HIT_COLOUR = (80, 255, 80)
MARK_COLOUR = (255, 200, 0)      # cyan-ish
VEH_COLOUR = (255, 60, 230)      # magenta (other vehicles)


@dataclass
class Settings:
    hfov_deg: float = 100.0
    cam_height_m: float = 1.3
    bonnet_frac: float = 0.0
    horizon_frac: float | None = None     # None = auto (vanishing point)
    lane_width_m: float | None = 3.5      # None = don't auto-calibrate scale
    grid_lines: int = 12
    grid_start_m: float = 1.0
    window_s: float = 3.0
    proc_width: int = 960
    speedup: float = 1.0                  # video is fast-forwarded by this factor (1 = real time)
    vehicles: bool = True                 # box other vehicles and show their speed in the video


def _ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class VideoWriter:
    """H.264 (browser-playable) via ffmpeg pipe; falls back to OpenCV mp4v."""

    def __init__(self, path: Path, w: int, h: int, fps: float):
        self.proc = None
        self.cv = None
        exe = _ffmpeg_exe()
        if exe:
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-", "-c:v", "libx264",
                   "-preset", "superfast", "-crf", "23", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(path)]
            flags = 0x08000000 if sys.platform == "win32" else 0   # CREATE_NO_WINDOW
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL, creationflags=flags)
        else:
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame):
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.cv.write(frame)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


def _resize(frame, width):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w / 2)) * 2), interpolation=cv2.INTER_LINEAR)


class FrameSource:
    """Decodes the video on a background thread (OpenCV releases the GIL while
    decoding/warping, so this overlaps with the analysis on the main thread)."""

    _END = object()

    def __init__(self, path: str, fn, maxsize: int = 16):
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.fn = fn
        self.path = path
        self.error = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        cap = cv2.VideoCapture(self.path)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self.q.put(self.fn(frame))
        except Exception as e:  # pragma: no cover - surfaced to the caller
            self.error = e
        finally:
            cap.release()
            self.q.put(self._END)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is self._END:
                if self.error:
                    raise self.error
                return
            yield item


def _detect_bonnet(gray_frames: list[np.ndarray]) -> float:
    """Fraction of the frame bottom that never changes (bonnet, dash, timestamp bar)."""
    if len(gray_frames) < 6:
        return 0.0
    st = np.stack(gray_frames).astype(np.float32).std(axis=0)
    H, W = st.shape
    row = np.median(st[:, int(W * 0.15): int(W * 0.85)], axis=1)
    ref = np.median(row[int(H * 0.55): int(H * 0.8)])
    if ref < 2.0:                 # whole video static (parked): can't tell
        return 0.0
    n = 0
    for y in range(H - 1, int(H * 0.6), -1):
        if row[y] < 0.25 * ref:
            n += 1
        else:
            break
    frac = n / H
    return float(min(frac + 0.01, 0.35)) if frac >= 0.02 else 0.0


def calibrate(path: str, s: Settings, log=print) -> tuple[CameraModel, dict]:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 300
    cap.release()
    idxs = np.linspace(0, max(n - 1, 0), num=min(16, max(n, 1))).astype(int)

    # seeking is slow in OpenCV, so spread the sample frames over a few decoders
    def grab(chunk):
        c = cv2.VideoCapture(path)
        got = []
        for i in chunk:
            c.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = c.read()
            if ok:
                got.append((int(i), _resize(fr, s.proc_width)))
        c.release()
        return got

    workers = max(1, min(4, os.cpu_count() or 1))
    chunks = [idxs[k::workers] for k in range(workers)]
    with ThreadPoolExecutor(workers) as ex:
        got = [g for part in ex.map(grab, chunks) for g in part]
    frames = [fr for _, fr in sorted(got, key=lambda x: x[0])]
    if not frames:
        raise RuntimeError("Could not read any frames from the video.")
    H, W = frames[0].shape[:2]
    info = {}
    if s.horizon_frac is not None:
        vp = (W / 2.0, s.horizon_frac * H)
        info["vanishing_point"] = "manual"
    else:
        vp = estimate_vanishing_point(frames)
        info["vanishing_point"] = "auto" if vp else "fallback"
        if vp is None:
            vp = (W / 2.0, 0.5 * H)
    bonnet = s.bonnet_frac
    if bonnet == 0:
        bonnet = _detect_bonnet([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
        if bonnet:
            info["bonnet_auto"] = round(bonnet, 3)
    cam = CameraModel(W, H, hfov_deg=s.hfov_deg, cam_height_m=s.cam_height_m, vp=vp,
                      bonnet_frac=bonnet)
    info["vp"] = [round(vp[0], 1), round(vp[1], 1)]

    if s.lane_width_m:
        widths = []
        road = RoadModel(cam, smooth=1.0)
        for fr in frames:
            bev = cam.warp_to_bev(fr)
            road.update(bev)
            mask, _ = detect_markings(bev, cam, road.mask)
            widths += measure_lane_spacing(mask, cam)
        if len(widths) >= 15:
            measured = float(np.median(widths))
            factor = s.lane_width_m / measured
            if 0.4 < factor < 2.5:
                cam.scale_height(factor)
                info["lane_width_measured_m"] = round(measured, 3)
                info["scale_factor"] = round(factor, 3)
    info["camera_height_m"] = round(cam.cam_height_m, 3)
    info["nearest_visible_m"] = round(cam.z_min, 2)
    log(f"calibration: {info}")
    return cam, info


def _draw_poly(img, pts, colour, thick):
    pts = np.asarray(pts)
    if len(pts) < 2:
        return
    ok = np.all(np.isfinite(pts), axis=1)
    pts = pts[ok]
    if len(pts) >= 2:
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, colour, thick, cv2.LINE_AA)


def process_video(path: str, out_dir: str, s: Settings | None = None,
                  progress: Callable[[float, str], None] | None = None, log=print,
                  preview: Callable[[np.ndarray], None] | None = None,
                  cancel: threading.Event | None = None) -> dict:
    s = s or Settings()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda p, m: None)

    # few cores: keep OpenCV from fighting the decoder thread and the encoder
    cv2.setNumThreads(1 if (os.cpu_count() or 2) <= 3 else 2)
    progress(0.01, "Calibrating camera (vanishing point, scale)")
    cam, calib = calibrate(path, s, log)

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 240:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    # Re-uploaded / converted videos often repeat frames (e.g. 25 fps shown as 30 fps by
    # repeating every 6th frame).  Repeated frames are skipped and real frames are timed
    # at the true rate, otherwise motion looks like jump-freeze-jump and timing fails.
    dup_rate = _duplicate_rate(path)
    fps_eff = _true_rate(fps, dup_rate) if dup_rate >= 0.04 else fps
    calib["repeated_frames"] = round(dup_rate, 3)
    if fps_eff != fps:
        calib["true_fps"] = round(fps_eff, 3)
    bev_size = cam.bev_size
    I2BEV = cam.I2BEV
    width = s.proc_width

    zoom_src = None              # full-resolution crop for far vehicles (x0, y0, x1, y1, scale)

    def prepare(frame):          # runs on the decoder thread (OpenCV releases the GIL)
        small = _resize(frame, width)
        bev = cv2.warpPerspective(small, I2BEV, bev_size, flags=cv2.INTER_LINEAR)
        if zoom_src is None:
            return small, bev, None
        x0, y0, x1, y1, k = zoom_src
        return small, bev, frame[int(y0 * k):int(y1 * k), int(x0 * k):int(x1 * k)]

    road = RoadModel(cam)
    grid_s = s.grid_start_m + np.arange(s.grid_lines) * 1.0
    tracker = MarkingTracker(grid_s)
    ego = EgoMotion(cam, fps)
    live = LiveTimeline(fps)
    odo = SurfaceOdometry(cam, fps_eff)
    gflow = GroundFlow(cam, fps_eff)
    lflow = LinePatternFlow(cam, fps_eff)
    recent_surf: deque = deque(maxlen=int(fps))
    writer = VideoWriter(out / "annotated.mp4", cam.width, cam.height, fps)

    # other vehicles: detected a few times per second, speed = ego speed + their speed
    # across the grid relative to us
    veh = det = None
    if s.vehicles:
        if MODEL_PATH.exists():
            det = VehicleDetector()
            model3d = None
            if Vehicle3DModel.available():
                try:
                    model3d = Vehicle3DModel()
                    calib["vehicle_model"] = "learned 3D boxes"
                except Exception as e:          # noqa: BLE001
                    log(f"vehicle3d model could not be loaded ({e}); using the built-in method")
            veh = VehicleSpeeds(cam, fps_eff, speedup=s.speedup, model=model3d)
            det_every = max(1, int(round(fps_eff / 4.0)))
            det_band = veh.band()
            zrect = veh.zoom_rect()
            src_w = cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_WIDTH) or cam.width
            zoom_src = (*zrect, src_w / cam.width)
            calib["vehicle_detection_hz"] = round(fps_eff / det_every, 2)
        else:
            log(f"vehicle model not found ({MODEL_PATH}); other vehicles are skipped")
    n_threads = cv2.getNumThreads()
    frame_t: list[float] = []             # true time of every frame (repeats share it)

    # Frames are drawn a few seconds after they are analysed, once their speed
    # values are final, so the video, the HUD and the table always agree.
    lag_raw = int(np.ceil(2 * live.half * fps)) + ego.enter + 6
    if veh is not None:                   # vehicle speeds look ~1 s ahead too
        lag_raw = max(lag_raw, int(np.ceil((1.5 * veh.half + 0.6) * fps)) + 2)
    pending: deque = deque()              # (frame index, small frame, overlay)
    rows_done: dict[int, dict] = {}
    st_hist = ego.history                 # per-frame 'stopped' flags (shared list)
    state = {"written": 0, "t_prev": 0.0}

    def hud_window_row(w):
        """Row of window ``w`` (0-based) computed from final values - same as the table."""
        if w in rows_done:
            return rows_done[w]
        a, b, fa, fb = window_bounds(w, s.window_s, 1e12, fps)
        if len(live.final) < fb:
            return None
        n_samp = bisect.bisect_left(live.ts, b) - bisect.bisect_left(live.ts, a)
        rows_done[w] = window_row(w, a, b, np.array(live.final[fa:fb]), np.array(st_hist[fa:fb]), n_samp,
                                  live.method[fa:fb])
        return rows_done[w]

    def ego_smooth(j):
        """Our speed at frame j for the vehicle speeds: a straight-line fit over the last
        second of final values (removes frame-to-frame jitter without lagging when we
        speed up or slow down); 0 when stopped."""
        if st_hist[j]:
            return 0.0
        a = max(0, j - int(round(fps)))
        seg = np.array(live.final[a:j + 1], float)
        k = np.flatnonzero(np.isfinite(seg))
        if len(k) < 3:
            return float(live.final[j])
        fit = np.polyfit(k - (j - a), seg[k], 1)
        return float(max(np.polyval(fit, 0.0), 0.0))

    def flush(upto_final):
        while pending and pending[0][0] < upto_final:
            j, small, ov = pending.popleft()
            t = j / fps
            v = live.final[j]
            if st_hist[j]:
                speed_txt = "speed    0.0 km/h (stopped)"
            else:
                src = {"M": " (markings)", "L": " (markings)", "S": " (surface)"}.get(live.method[j], "")
                speed_txt = "speed  " + (f"{v * 3.6:5.1f} km/h{src}" if np.isfinite(v) else "  --")
            w = int(t // s.window_s)
            row = hud_window_row(w - 1) if w >= 1 else None
            if row is None:
                win_txt = "window --"
            elif row["speed_kmh"] is None:
                win_txt = f"window {row['window']}: --"
            else:
                win_txt = f"window {row['window']}: {row['speed_kmh']:5.1f} km/h"
            vehs = None
            if veh is not None:
                vehs = veh.boxes_at(j, frame_t[j], ego_smooth(j), hold_frames=2 * det_every)
            vis = _render(small, ov, [f"t = {t:6.2f} s", speed_txt, win_txt], vehs)
            writer.write(vis)
            state["written"] += 1
            if preview is not None and _time.monotonic() - state["t_prev"] > 0.2:
                state["t_prev"] = _time.monotonic()
                preview(vis)

    marking_mask = None
    marking_wide = None
    fi = 0
    flash: dict[int, float] = {}
    free_s = None                     # unobstructed road length ahead (m)
    dedup = fps_eff != fps
    sp = max(float(s.speedup or 1.0), 0.01)
    sig_prev = None
    diffs: deque = deque(maxlen=30)
    n_unique = 0
    last_ov = None
    try:
        for small, bev, zoom_img in FrameSource(path, prepare):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Cancelled")

            # repeated frame?  (only meaningful while the picture is actually changing)
            sig = cv2.resize(cv2.cvtColor(small[small.shape[0] // 2:], cv2.COLOR_BGR2GRAY), (160, 45),
                             interpolation=cv2.INTER_AREA).astype(np.float32)
            repeat = False
            if sig_prev is not None:
                dif = float(np.mean(np.abs(sig - sig_prev)))
                med = float(np.median(diffs)) if len(diffs) >= 5 else dif
                repeat = dedup and med > 1.5 and dif < 0.15 * med
                if not repeat:
                    diffs.append(dif)
            sig_prev = sig
            if repeat and last_ov is not None:
                # nothing new in this frame: keep the per-frame records aligned, skip analysis
                live.surface.append(np.nan)
                live.lines.append(np.nan)
                ego.history.append(ego.stationary)
                pending.append((fi, small, last_ov))
                frame_t.append(frame_t[-1] if frame_t else 0.0)
                fi += 1
                ready = fi - 1 - lag_raw
                if ready >= 0:
                    live.compute_raw(ready, st_hist)
                    flush(live.finalize(ready))
                continue
            t = n_unique / fps_eff            # true time of this real frame
            if veh is not None and n_unique % det_every == 0:
                cv2.setNumThreads(os.cpu_count() or 2)    # the detector wants every core
                y0, y1 = det_band
                full = [(b + [0, y0, 0, y0], c, k) for b, c, k in det.detect(small[y0:y1])]
                zx0, zy0, _, _, zk = zoom_src
                zoom = [(b / zk + [zx0, zy0, zx0, zy0], c, k) for b, c, k in det.detect(zoom_img)]
                veh.observe(fi, t, veh.merge(full, zoom, zrect), frame=small)
                cv2.setNumThreads(n_threads)
            n_unique += 1

            road.update(bev, marking_mask, line_mask=marking_wide)
            # look for paint well beyond the current road edges too, so solid edge lines
            # outside a too-narrow colour estimate can still be found (and snapped to)
            wide = road.band_mask(margin_m=5.0)
            band = road.band_mask()
            marking_wide, ends, tophat = detect_markings(bev, cam, wide if wide is not None else road.mask,
                                                         stats_mask=band, return_tophat=True)
            marking_mask = cv2.bitwise_and(marking_wide, band) if band is not None else marking_wide
            grid = RoadGrid(cam, road, s.grid_start_m, s.grid_lines)
            f = grid.free_distance(road.mask)
            # drop instantly when something pulls in front, recover gently
            free_s = f if (free_s is None or f < free_s) else 0.85 * free_s + 0.15 * f

            # only keep marking ends that lie on the road (with a small margin)
            dets = []
            if ends:
                rc = grid.to_road_coords(np.array([[e.x, e.y] for e in ends]))
                for e, (ss, dd) in zip(ends, rc):
                    hw = np.interp(ss, grid.s, grid.half_w) / cam.ppm
                    # on the road, and not on a vehicle ahead (white bodywork looks like paint)
                    if abs(dd) <= hw + 0.6 and (free_s is None or ss < free_s - 0.5):
                        dets.append((float(ss), float(dd), e.kind, e.x, e.y))
            # dashed-line pattern motion (robust when dashes move far between frames)
            lflow.observe_dashes([d[1] for d in dets])
            d_line = lflow.update(tophat, grid, free_s)
            v_line = d_line * fps_eff if np.isfinite(d_line) else np.nan
            live.lines.append(v_line / sp if np.isfinite(v_line) else np.nan)
            # road-surface speed (works with or without markings)
            d_strip = odo.update(small, free_s)          # dense: road texture just ahead
            d_feat = gflow.update(small)                 # sparse: features over all ground
            v_surf = _combine(d_strip * fps_eff, d_feat * fps_eff, recent_surf)
            if np.isfinite(v_surf):
                recent_surf.append(v_surf)
            live.surface.append(v_surf / sp)          # on-screen -> real speed
            marking_moving = any(tr.hits >= 5 and tr.misses == 0 and -tr.x[1] > 2.0
                                 for tr in tracker.tracks)
            marking_moving = (marking_moving or (np.isfinite(v_surf) and v_surf > 2.0)
                              or lflow.moving)
            mature = [tr for tr in tracker.tracks if tr.hits >= 10 and tr.misses == 0]
            markings_still = bool(mature) and all(abs(tr.x[1]) < 0.4 for tr in mature)
            tracker.stationary = ego.update(bev, marking_moving, markings_still, gflow.image_motion)
            live.add([SpeedSample(x.t, x.speed_mps / sp, x.track, x.line_pair, x.dt)
                      for x in tracker.step(t, dets)])
            for tc, k in tracker.recent_crossings:
                flash[k] = tc
            tracker.recent_crossings.clear()

            last_ov = _overlay(cam, grid, marking_mask, tracker, t, flash, free_s)
            pending.append((fi, small, last_ov))
            frame_t.append(t)
            fi += 1
            ready = fi - 1 - lag_raw
            if ready >= 0:
                live.compute_raw(ready, st_hist)
                flush(live.finalize(ready))
            if fi % 10 == 0:
                p = 0.03 + 0.96 * (fi / total if total else 0.5)
                progress(min(p, 0.99), f"Processing frame {fi}/{total or '?'}")

        # end of video: everything is known now
        live.compute_raw(fi - 1, st_hist)
        flush(live.finalize(fi - 1, complete=True))
    finally:
        writer.close()

    duration = fi / fps
    stationary = np.array(st_hist[:fi], bool)
    timeline = np.array(live.final[:fi], float)
    rows = aggregate_windows(tracker.samples, timeline, stationary, fps, s.window_s, live.method[:fi])
    moving = [r["speed_kmh"] for r in rows if r["speed_kmh"]]
    if moving and np.median(moving) > 200 and sp == 1.0:
        calib["warning"] = (f"On-screen speed is about {np.median(moving):.0f} km/h - this video looks "
                            "fast-forwarded. If you know by how much (e.g. 3x), set 'Video sped up' "
                            "in Settings and analyse again.")

    result = {
        "fps": round(fps, 3), "frames": fi, "duration_s": round(duration, 2),
        "calibration": calib, "settings": asdict(s), "windows": rows,
        "vehicles": veh.summary(lambda tt: ego_smooth(min(bisect.bisect_left(frame_t, tt), fi - 1)))
                    if veh is not None else [],
        "stationary_s": round(float(stationary.sum()) / fps, 2),
        "surface_kmh": [None if not np.isfinite(v) else round(float(v) * 3.6, 1) for v in live.surface[:fi]],
        "timeline_kmh": [None if np.isnan(v) else round(float(v) * 3.6, 1) for v in timeline],
        "samples": [{"t": round(x.t, 3), "speed_kmh": round(x.speed_mps / sp * 3.6, 2),
                     "track": x.track, "lines": list(x.line_pair), "dt_s": round(x.dt, 4)}
                    for x in tracker.samples],
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    if veh is not None and os.environ.get("RS_DUMP_VEH"):        # debugging aid
        import pickle
        (out / "vehicles.pkl").write_bytes(pickle.dumps(veh))
    with open(out / "speed_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window", "start_s", "end_s", "speed_kmh", "speed_mps", "status", "method",
                    "marking_samples"])
        for r in rows:
            w.writerow([r["window"], r["start_s"], r["end_s"], r["speed_kmh"], r["speed_mps"],
                        r["status"], r["method"], r["samples"]])
    progress(1.0, "Done")
    return result


def _duplicate_rate(path: str, n: int = 300) -> float:
    """Fraction of frames that exactly repeat the previous one (frame-rate conversion)."""
    cap = cv2.VideoCapture(path)
    prev, diffs = None, []
    for _ in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(fr[fr.shape[0] // 2:], cv2.COLOR_BGR2GRAY), (160, 45),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(g - prev))))
        prev = g
    cap.release()
    d = np.array(diffs)
    if len(d) < 20:
        return 0.0
    med = float(np.median(d))
    if med < 1.5:                 # mostly standing still: can't tell repeats from stillness
        return 0.0
    return float((d < 0.15 * med).mean())


def _true_rate(fps: float, dup_rate: float) -> float:
    eff = fps * (1.0 - dup_rate)
    for std in (23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0):
        if abs(eff - std) / std < 0.03:
            return std
    return eff


def _combine(v_strip: float, v_feat: float, recent) -> float:
    """Merge the two road-surface speed estimates (m/s) for one frame.

    The dense strip match is the more accurate one (unbiased on clear video, a few %
    low on very blurry/compressed video); feature tracking covers more of the picture
    and fills the frames where the strip has nothing to lock onto.
    """
    ok_s = np.isfinite(v_strip) and v_strip > 0
    ok_f = np.isfinite(v_feat) and v_feat > 0
    ref = float(np.median(recent)) if len(recent) >= 5 else None
    if ok_s and (ref is None or abs(v_strip - ref) <= 0.3 * ref or not ok_f):
        return v_strip
    if ok_f and (ref is None or abs(v_feat - ref) <= 0.3 * ref):
        return v_feat
    if ok_s and ok_f and abs(v_strip - v_feat) <= 0.15 * max(v_strip, v_feat):
        return v_strip                        # both agree on a real change of speed
    return np.nan


def _overlay(cam, grid: RoadGrid, mmask, tracker: MarkingTracker, t, flash, free_s=None):
    """Everything to draw for one frame, in image coordinates (compact)."""
    ov = {"marks": [], "polys": [], "dots": []}
    if mmask is not None:
        cnts, _ = cv2.findContours(mmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:                    # contours found in the BEV, mapped into the frame
            p = cv2.perspectiveTransform(c.astype(np.float32), cam.BEV2I)
            if np.all(np.isfinite(p)):
                ov["marks"].append(np.round(p).astype(np.int32))

    far = grid.line_s[-1] + 0.5
    if free_s is not None:            # don't paint the grid over a vehicle ahead
        far = min(far, max(free_s, grid.line_s[0] + 0.5))
    for side in ("left", "right"):
        ov["polys"].append((cam.bev_to_image(grid.edge_bev(side, far)[::4]), EDGE_COLOUR, 3))
    for k, sk in enumerate(grid.line_s):
        rung = grid.rung_bev(sk, samples=8)
        if rung is None or sk > far:
            continue
        hit = k in flash and (t - flash[k]) < 1.5 / 30
        ov["polys"].append((cam.bev_to_image(rung), HIT_COLOUR if hit else GRID_COLOUR, 3 if hit else 2))

    for tr in tracker.tracks:
        if tr.misses or tr.hits < 3 or tr.x[0] > min(grid.line_s[-1] + 1.0, far):
            continue
        p = cam.bev_to_image(np.array([tr.bev_xy]))[0]
        if np.all(np.isfinite(p)):
            ov["dots"].append((int(p[0]), int(p[1])))
    return ov


_TINT_CACHE: dict = {}


def _render(frame, ov, hud_lines, vehicles=None):
    vis = frame.copy()
    if len(ov["marks"]):
        h, w = frame.shape[:2]
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, ov["marks"], -1, 255, -1)
        x, y, bw, bh = cv2.boundingRect(m)
        if bw and bh:
            roi = vis[y:y + bh, x:x + bw]
            key = (bh, bw)
            if key not in _TINT_CACHE:
                _TINT_CACHE.clear()
                _TINT_CACHE[key] = np.full((bh, bw, 3), MARK_COLOUR, np.uint8)
            tinted = cv2.addWeighted(roi, 0.45, _TINT_CACHE[key], 0.55, 0)
            cv2.copyTo(tinted, m[y:y + bh, x:x + bw], roi)
    for pts, colour, thick in ov["polys"]:
        _draw_poly(vis, pts, colour, thick)
    for c in ov["dots"]:
        cv2.circle(vis, c, 4, (0, 0, 255), -1, cv2.LINE_AA)
    if vehicles:
        _draw_vehicles(vis, frame, vehicles)
    cv2.rectangle(vis, (10, 10), (330, 92), (20, 20, 20), -1)
    for i, txt in enumerate(hud_lines):
        cv2.putText(vis, txt, (20, 34 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return vis


def _draw_vehicles(vis, frame, vehicles):
    """Rectangle round each vehicle with its absolute speed on a tag above it.

    The grid / marking overlay is wiped inside each box first so the vehicle stays
    clearly visible, and tags are moved so they never overlap (near vehicles first)."""
    H, W = vis.shape[:2]
    items = []
    for box, kmh, dist, tid in vehicles:
        x1, y1, x2, y2 = [int(round(c)) for c in box]
        x1, x2 = max(x1, 0), min(x2, W - 1)
        y1, y2 = max(y1, 0), min(y2, H - 1)
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            items.append((x1, y1, x2, y2, kmh, dist if np.isfinite(dist) else 1e9))
    for x1, y1, x2, y2, _, _ in items:
        vis[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    for x1, y1, x2, y2, _, _ in items:
        cv2.rectangle(vis, (x1, y1), (x2, y2), VEH_COLOUR, 2, cv2.LINE_AA)
    placed = []
    for x1, y1, x2, y2, kmh, _ in sorted(items, key=lambda v: v[5]):     # nearest first
        if not np.isfinite(kmh):
            continue
        txt = f"{abs(kmh):.0f} km/h" + (" oncoming" if kmh < -5 else "")
        (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lw, lh = tw + 8, th + base + 6
        tx = min(max(x1, 0), W - lw)
        ty = y1 - lh - 2                       # top-left of the tag, just above the box
        for _ in range(6):                     # move up past tags already placed
            hit = [p for p in placed if tx < p[2] and tx + lw > p[0] and ty < p[3] and ty + lh > p[1]]
            if not hit:
                break
            ty = min(p[1] for p in hit) - lh - 2
        if ty < 0:
            ty = y2 + 2
        placed.append((tx, ty, tx + lw, ty + lh))
        cv2.rectangle(vis, (tx, ty), (tx + lw, ty + lh), VEH_COLOUR, -1)
        cv2.putText(vis, txt, (tx + 4, ty + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
