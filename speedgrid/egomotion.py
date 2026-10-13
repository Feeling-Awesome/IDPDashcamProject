"""Stationary-vehicle detection from the road surface right in front of the car.

The near-field road patch of the bird's-eye view is high-pass filtered and
registered against the previous frame with FFT **phase correlation**.  When
the car stands still the patch is identical up to noise: the correlation peak
is strong and sits at zero shift.  When the car moves, the asphalt texture
shifts by v/fps metres per frame (or blurs, weakening the peak).

A hysteresis on top (N consecutive still frames to enter "stopped", a few
moving frames to leave it) avoids flicker, and moving lane-marking tracks
always veto a "stopped" decision (protects against static overlays such as a
dashcam timestamp lying in the patch).
"""
from __future__ import annotations

import cv2
import numpy as np

from .camera import CameraModel


class EgoMotion:
    def __init__(self, cam: CameraModel, fps: float, enter_frames: int | None = None,
                 exit_frames: int = 3):
        self.cam = cam
        self.fps = fps
        Wb, Hb = cam.bev_size
        ppm = cam.ppm
        valid = cam.valid_bev_mask()
        # ego-lane patch: 0.3 .. 7 m ahead, +-1.8 m laterally, shrunk to the visible area
        y0, y1 = int(Hb - 7.0 * ppm), int(Hb - 0.3 * ppm)
        x0, x1 = int(Wb / 2 - 1.8 * ppm), int(Wb / 2 + 1.8 * ppm)
        while y1 - y0 > 40 and valid[y1 - 1, x0:x1].min() == 0:
            y1 -= 2
        self.box = (x0, x1, y0, y1)
        h, w = y1 - y0, x1 - x0
        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        self.prev = None
        self.enter = enter_frames or max(int(round(0.3 * fps)), 4)
        self.exit = exit_frames
        self.still_run = 0
        self.move_run = 0
        self.stationary = False
        self.history: list[bool] = []          # final per-frame decision
        self.shift_m: list[float] = []

    def _patch(self, bev_bgr):
        x0, x1, y0, y1 = self.box
        g = cv2.cvtColor(bev_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        hp = g - cv2.GaussianBlur(g, (0, 0), 3.0)     # keep texture, drop shading
        return hp

    def update(self, bev_bgr, marking_moving: bool, markings_still: bool = False,
               image_motion: float = float("nan")) -> bool:
        cur = self._patch(bev_bgr)
        still = False
        shift = np.nan
        if self.prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(self.prev, cur, self.win)
            shift = abs(dy) / self.cam.ppm
            # also require the frames to actually look alike (identical up to noise)
            diff = float(np.mean(np.abs(cur - self.prev)))
            energy = float(np.mean(np.abs(cur))) + 1e-3
            # phaseCorrelate has a ~0.5 px bias at zero motion, hence the 1 px tolerance;
            # the frame-difference test is the decisive one (texture identical up to noise)
            still = abs(dy) <= 1.0 and abs(dx) <= 1.0 and resp >= 0.15 and diff < 0.35 * energy
        self.prev = cur
        self.shift_m.append(shift)

        # lane markings that are tracked but not moving are independent evidence
        still = still or markings_still
        # ...but the picture as a whole must not be moving: tracked feature points
        # across the ground/roadside barely move when the car really stands still.
        # (Blurry or heavily compressed video can make the road patch look static.)
        if np.isfinite(image_motion):
            if image_motion > 0.6:
                still = False
            elif image_motion < 0.25 and markings_still:
                still = True
        if still and not marking_moving:
            self.still_run += 1
            self.move_run = 0
        else:
            self.move_run += 1
            self.still_run = 0
        if not self.stationary and self.still_run >= self.enter:
            self.stationary = True
            # the frames that built up the evidence were stationary too
            for i in range(1, self.enter):
                if len(self.history) >= i:
                    self.history[-i] = True
        elif self.stationary and self.move_run >= self.exit:
            self.stationary = False
        self.history.append(self.stationary)
        return self.stationary
