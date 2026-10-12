"""
Dashcam Speed Detector — CV Processor
======================================
Pipeline:
  1. PerspectiveTransformer  — camera view ↔ bird's-eye view (BEV)
  2. LaneLineDetector        — Hough-based left/right lane detection for auto-calibration
  3. GridManager             — perspective-aware grid (2 vertical + N horizontal @ 1m each)
  4. LaneMarkingTracker      — sparse optical-flow tracking of marking blobs in BEV;
                               records horizontal-gridline crossing timestamps → speed
  5. VideoProcessor          — orchestrates all components, exposes frame iterator & table
"""

import cv2
import numpy as np
import threading
import time
from collections import deque
from typing import List, Tuple, Optional, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# 1. Perspective Transformer
# ─────────────────────────────────────────────────────────────────────────────

class PerspectiveTransformer:
    """
    Inverse Perspective Mapping (IPM) — maps a dashcam frame to a top-down
    bird's-eye view so that real-world distances (metres) map to pixel counts.

    BEV coordinate system
    ─────────────────────
      • Origin: top-left
      • y increases downward (away from camera in the real world becomes
        the TOP of BEV, so lane markings move DOWN toward higher y as the
        car moves forward)
    """

    BEV_W = 500
    BEV_H = 700

    # Left / right edge x-positions in BEV (10 % margin on each side)
    BEV_LEFT_X  = int(BEV_W * 0.10)
    BEV_RIGHT_X = int(BEV_W * 0.90)

    def __init__(self, frame_w: int, frame_h: int):
        self.fw, self.fh = frame_w, frame_h
        self.calibrated = False
        self.M: Optional[np.ndarray] = None
        self.M_inv: Optional[np.ndarray] = None
        self._set_default_transform()

    # ── default (un-calibrated) transform ────────────────────────────────────

    def _set_default_transform(self):
        fw, fh = self.fw, self.fh
        src = np.float32([
            [fw * 0.10, fh * 0.97],   # bottom-left
            [fw * 0.90, fh * 0.97],   # bottom-right
            [fw * 0.58, fh * 0.60],   # top-right (near horizon)
            [fw * 0.42, fh * 0.60],   # top-left  (near horizon)
        ])
        bw, bh = self.BEV_W, self.BEV_H
        dst = np.float32([
            [self.BEV_LEFT_X,  bh],
            [self.BEV_RIGHT_X, bh],
            [self.BEV_RIGHT_X, 0],
            [self.BEV_LEFT_X,  0],
        ])
        self._apply(src, dst)

    def _apply(self, src: np.ndarray, dst: np.ndarray):
        self.M     = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    # ── calibration from Hough-detected lane lines ───────────────────────────

    def calibrate(self, left_line, right_line):
        """
        Refine the transform using detected left / right lane lines.
        Each line is (x1, y1, x2, y2).
        """
        if left_line is None or right_line is None:
            return

        # Vanishing point
        vp = _line_intersection(left_line, right_line)
        if vp is None:
            return
        vp_x, vp_y = vp
        if not (0 < vp_y < self.fh * 0.75):
            return

        # Bottom x positions (extrapolated to image bottom)
        lx_bot = _extrapolate_x(left_line,  self.fh)
        rx_bot = _extrapolate_x(right_line, self.fh)
        if lx_bot is None or rx_bot is None:
            return

        half_w = (rx_bot - lx_bot) * 0.04  # tiny horizontal spread at VP
        y_top  = vp_y + 15

        src = np.float32([
            [lx_bot,          self.fh],
            [rx_bot,          self.fh],
            [vp_x + half_w,   y_top],
            [vp_x - half_w,   y_top],
        ])
        bw, bh = self.BEV_W, self.BEV_H
        dst = np.float32([
            [self.BEV_LEFT_X,  bh],
            [self.BEV_RIGHT_X, bh],
            [self.BEV_RIGHT_X, 0],
            [self.BEV_LEFT_X,  0],
        ])
        self._apply(src, dst)
        self.calibrated = True

    # ── transforms ───────────────────────────────────────────────────────────

    def warp_to_bev(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.M, (self.BEV_W, self.BEV_H))

    def pts_to_bev(self, pts: np.ndarray) -> np.ndarray:
        if pts is None or len(pts) == 0:
            return np.empty((0, 2), np.float32)
        return cv2.perspectiveTransform(
            pts.reshape(-1, 1, 2).astype(np.float32), self.M).reshape(-1, 2)

    def pts_to_cam(self, pts: np.ndarray) -> np.ndarray:
        if pts is None or len(pts) == 0:
            return np.empty((0, 2), np.float32)
        return cv2.perspectiveTransform(
            pts.reshape(-1, 1, 2).astype(np.float32), self.M_inv).reshape(-1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Helper geometry utilities
# ─────────────────────────────────────────────────────────────────────────────

def _line_intersection(l1, l2) -> Optional[Tuple[float, float]]:
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-8:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)

def _extrapolate_x(line, y_target) -> Optional[float]:
    x1, y1, x2, y2 = line
    dy = y2 - y1
    if abs(dy) < 1e-8:
        return None
    return x1 + (y_target - y1) / dy * (x2 - x1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lane Line Detector (for transform calibration)
# ─────────────────────────────────────────────────────────────────────────────

class LaneLineDetector:
    """
    Detects the dominant left and right lane boundary lines using:
      Grayscale → Gaussian blur → Canny → ROI mask → HoughLinesP → regression
    """

    def __init__(self, fw: int, fh: int):
        self.fw, self.fh = fw, fh

    def detect(self, frame: np.ndarray):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        masked = cv2.bitwise_and(edges, self._roi(edges))

        lines = cv2.HoughLinesP(masked, 1, np.pi / 180, threshold=30,
                                minLineLength=50, maxLineGap=120)
        if lines is None:
            return None, None

        return self._separate(lines)

    def _roi(self, img: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(img)
        fw, fh = self.fw, self.fh
        poly = np.array([[
            (int(fw * 0.00), fh),
            (int(fw * 0.44), int(fh * 0.58)),
            (int(fw * 0.56), int(fh * 0.58)),
            (int(fw * 1.00), fh),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, poly, 255)
        return mask

    def _separate(self, lines):
        left_pts,  right_pts  = [], []
        mid = self.fw / 2

        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            dx = x2 - x1
            if abs(dx) < 1:
                continue
            slope = (y2 - y1) / dx
            if abs(slope) < 0.3 or abs(slope) > 8:
                continue
            cx = (x1 + x2) / 2
            if slope < 0 and cx < mid:
                left_pts.extend([(x1, y1), (x2, y2)])
            elif slope > 0 and cx > mid:
                right_pts.extend([(x1, y1), (x2, y2)])

        left  = self._fit(left_pts)
        right = self._fit(right_pts)
        return left, right

    def _fit(self, pts) -> Optional[Tuple]:
        if len(pts) < 4:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        try:
            coeffs = np.polyfit(ys, xs, 1)   # x = a*y + b
        except np.linalg.LinAlgError:
            return None
        y_bot = self.fh
        y_top = int(self.fh * 0.58)
        x_bot = int(np.polyval(coeffs, y_bot))
        x_top = int(np.polyval(coeffs, y_top))
        return (x_bot, y_bot, x_top, y_top)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Grid Manager
# ─────────────────────────────────────────────────────────────────────────────

class GridManager:
    """
    Defines the perspective-aware grid entirely in BEV coordinates, then
    projects it back to the camera frame for display.

    Grid definition (BEV space)
    ───────────────────────────
      • Two vertical guidelines at x = BEV_LEFT_X and x = BEV_RIGHT_X
        (these map to the left / right road edges in camera view).
      • Horizontal lines every `pixels_per_meter` pixels in y
        (= 1 m in the real world, given the lane-width calibration).
    """

    GRID_COLOR   = (0, 220, 255)   # cyan-yellow (stands out on road)
    EDGE_COLOR   = (0, 220, 255)
    LABEL_COLOR  = (0, 220, 255)
    GRID_THICK   = 1
    EDGE_THICK   = 2

    def __init__(self, lane_width_m: float = 3.5):
        bw = PerspectiveTransformer.BEV_W
        bh = PerspectiveTransformer.BEV_H
        lx = PerspectiveTransformer.BEV_LEFT_X
        rx = PerspectiveTransformer.BEV_RIGHT_X

        self.bev_w, self.bev_h = bw, bh
        self.left_x, self.right_x = lx, rx

        lane_px = rx - lx
        self.pixels_per_meter: float = lane_px / lane_width_m

        # Horizontal gridline y-positions in BEV (0 = bottom, increasing = up)
        # We store them as actual BEV y values (origin top-left, so bottom = bh)
        self.h_lines_bev: List[int] = []
        y = float(bh)
        while y >= 0:
            self.h_lines_bev.append(int(round(y)))
            y -= self.pixels_per_meter

    # ── drawing ───────────────────────────────────────────────────────────────

    def draw_on_frame(self, frame: np.ndarray,
                      tf: PerspectiveTransformer) -> np.ndarray:
        out = frame.copy()
        bh  = self.bev_h
        lx, rx = self.left_x, self.right_x

        # ── vertical edge guidelines (sampled, so curves follow perspective) ──
        num_v = 20
        for edge_x in (lx, rx):
            ys  = np.linspace(0, bh, num_v)
            pts_bev = np.column_stack([np.full(num_v, edge_x), ys])
            pts_cam = tf.pts_to_cam(pts_bev).astype(np.int32)
            pts_cam = self._clip_pts(pts_cam, frame.shape)
            cv2.polylines(out, [pts_cam.reshape(-1, 1, 2)], False,
                          self.EDGE_COLOR, self.EDGE_THICK, cv2.LINE_AA)

        # ── horizontal gridlines (sampled across width for road-curve follow) ─
        num_h = 15
        xs = np.linspace(lx, rx, num_h)
        for idx, y_bev in enumerate(self.h_lines_bev):
            pts_bev = np.column_stack([xs, np.full(num_h, y_bev)])
            pts_cam = tf.pts_to_cam(pts_bev).astype(np.int32)
            pts_cam = self._clip_pts(pts_cam, frame.shape)

            thick = self.EDGE_THICK if idx == 0 else self.GRID_THICK
            cv2.polylines(out, [pts_cam.reshape(-1, 1, 2)], False,
                          self.GRID_COLOR, thick, cv2.LINE_AA)

            # Distance label (skip 0 m label)
            if 0 < idx <= 12:
                label_pt = pts_cam[0]
                lx_c, ly_c = int(label_pt[0]) + 5, int(label_pt[1]) - 4
                if 0 <= lx_c < frame.shape[1] and 0 <= ly_c < frame.shape[0]:
                    cv2.putText(out, f"{idx}m", (lx_c, ly_c),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                self.LABEL_COLOR, 1, cv2.LINE_AA)
        return out

    @staticmethod
    def _clip_pts(pts: np.ndarray, shape) -> np.ndarray:
        h, w = shape[:2]
        pts[:, 0] = np.clip(pts[:, 0], -w, 2 * w)
        pts[:, 1] = np.clip(pts[:, 1], -h, 2 * h)
        return pts


# ─────────────────────────────────────────────────────────────────────────────
# 4. Lane Marking Tracker
# ─────────────────────────────────────────────────────────────────────────────

class LaneMarkingTracker:
    """
    Tracks lane marking blobs in BEV using sparse Lucas-Kanade optical flow.
    
    Speed measurement
    ─────────────────
    As the car moves forward, dashed lane markings travel downward in BEV
    (y increases).  When a tracked feature point crosses a horizontal gridline,
    we record its timestamp.  Consecutive crossings of adjacent gridlines
    (spaced exactly 1 m apart) yield:

        speed [m/s] = 1 / Δt
        speed [km/h] = 3.6 / Δt
    """

    MAX_FEATURES   = 80
    REDETECT_EVERY = 12   # frames

    LK_PARAMS = dict(
        winSize    = (21, 21),
        maxLevel   = 3,
        criteria   = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )

    def __init__(self, grid: GridManager):
        self.grid = grid
        self.prev_gray: Optional[np.ndarray]  = None
        self.tracked_pts: Optional[np.ndarray] = None   # (N,1,2) float32
        self.pt_ids: List[int] = []
        self._next_id = 0

        # {point_id: {line_index: timestamp}}
        self.crossings: Dict[int, Dict[int, float]] = {}
        # Raw speed samples: (timestamp, speed_kmh)
        self.samples: deque = deque(maxlen=500)

        self._frame_idx = 0
        self._fps = 30.0

    # ── public API ─────────────────────────────────────────────────────────

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, v):
        self._fps = max(1.0, float(v))

    def update(self, cam_frame: np.ndarray, tf: PerspectiveTransformer,
               timestamp: float) -> np.ndarray:
        """
        Process one frame.  Returns the BEV image (for debug / overlay use).
        """
        bev      = tf.warp_to_bev(cam_frame)
        gray_bev = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self._init_tracking(bev)
        else:
            self._optical_flow(gray_bev, timestamp)
            if self._frame_idx % self.REDETECT_EVERY == 0:
                self._redetect(bev)

        self.prev_gray  = gray_bev
        self._frame_idx += 1
        return bev

    def get_speed_kmh(self, window_s: float, current_time: float
                      ) -> Optional[float]:
        """Median speed over the last `window_s` seconds; None if no data."""
        recent = [s for t, s in self.samples
                  if current_time - t <= window_s]
        if len(recent) < 2:
            return None
        return float(np.median(recent))

    def draw_on_frame(self, frame: np.ndarray,
                      tf: PerspectiveTransformer) -> np.ndarray:
        """Overlay tracked marking dots on the camera-view frame."""
        if self.tracked_pts is None or len(self.tracked_pts) == 0:
            return frame
        pts_cam = tf.pts_to_cam(
            self.tracked_pts.reshape(-1, 2)).astype(np.int32)
        h, w = frame.shape[:2]
        for (x, y) in pts_cam:
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 4, (0, 100, 255), -1, cv2.LINE_AA)
        return frame

    # ── internal helpers ──────────────────────────────────────────────────

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _init_tracking(self, bev: np.ndarray):
        pts = self._detect_features(bev)
        if pts is not None and len(pts) > 0:
            self.tracked_pts = pts
            self.pt_ids = [self._new_id() for _ in range(len(pts))]

    def _optical_flow(self, gray_bev: np.ndarray, timestamp: float):
        if self.tracked_pts is None or len(self.tracked_pts) == 0:
            return

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray_bev,
            self.tracked_pts.astype(np.float32), None, **self.LK_PARAMS)

        if new_pts is None:
            return

        mask    = status.flatten() == 1
        old_pts = self.tracked_pts.reshape(-1, 2)[mask]
        new_pts = new_pts.reshape(-1, 2)[mask]
        ids     = [pid for pid, ok in zip(self.pt_ids, mask) if ok]

        self._check_crossings(ids, old_pts, new_pts, timestamp)

        # Keep only points still inside BEV
        bh, bw = self.grid.bev_h, self.grid.bev_w
        valid, valid_ids = [], []
        for pid, pt in zip(ids, new_pts):
            x, y = pt
            if 0 <= x <= bw and 0 <= y <= bh:
                valid.append(pt)
                valid_ids.append(pid)

        if valid:
            self.tracked_pts = np.array(valid, np.float32).reshape(-1, 1, 2)
            self.pt_ids      = valid_ids
        else:
            self.tracked_pts = None
            self.pt_ids      = []

        # Prune stale crossings for lost points
        alive = set(self.pt_ids)
        for pid in list(self.crossings.keys()):
            if pid not in alive:
                del self.crossings[pid]

    def _check_crossings(self, ids, old_pts, new_pts, timestamp: float):
        """
        Detect downward crossings of each horizontal gridline.
        Lane markings move downward (y ↑) as the car moves forward.
        """
        h_lines = self.grid.h_lines_bev  # sorted: index 0 = bottom (y = bev_h)
        dt_frame = 1.0 / self._fps

        for pid, old_pt, new_pt in zip(ids, old_pts, new_pts):
            old_y, new_y = old_pt[1], new_pt[1]

            # Only count downward motion
            if new_y <= old_y:
                continue

            for li, y_line in enumerate(h_lines):
                # Downward crossing: old_y was above the line, new_y is at or below
                if old_y < y_line <= new_y:
                    # Interpolate exact crossing time
                    span = new_y - old_y
                    frac = (y_line - old_y) / span if span > 0 else 0.5
                    t_cross = timestamp - (1.0 - frac) * dt_frame

                    if pid not in self.crossings:
                        self.crossings[pid] = {}

                    # Speed from previous adjacent gridline crossing
                    prev_li = li - 1   # one line closer to horizon = 1 m farther
                    if prev_li in self.crossings.get(pid, {}):
                        dt = t_cross - self.crossings[pid][prev_li]
                        if 0.05 < dt < 10.0:        # 0.36 – 72 km/h sanity window
                            speed_kmh = 3.6 / dt    # 1 m / dt seconds * 3.6
                            if 1.0 < speed_kmh < 280.0:
                                self.samples.append((timestamp, speed_kmh))

                    self.crossings[pid][li] = t_cross

    def _detect_features(self, bev: np.ndarray) -> Optional[np.ndarray]:
        """Detect good features to track on lane marking pixels."""
        mask = self._marking_mask(bev)
        pts  = cv2.goodFeaturesToTrack(
            mask,
            maxCorners  = self.MAX_FEATURES,
            qualityLevel= 0.15,
            minDistance = 14,
            blockSize   = 7,
        )
        return pts

    def _marking_mask(self, bev: np.ndarray) -> np.ndarray:
        """Binary mask of likely lane marking pixels in BEV."""
        hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)

        # White markings: high lightness
        white  = cv2.inRange(hls,
                             np.array([0,  170, 0]),
                             np.array([255, 255, 255]))
        # Yellow markings
        yellow = cv2.inRange(hls,
                             np.array([12, 80,  80]),
                             np.array([38, 255, 255]))
        mask = cv2.bitwise_or(white, yellow)

        # Restrict to lane area
        lm = np.zeros(mask.shape, np.uint8)
        lm[:, self.grid.left_x: self.grid.right_x] = 255
        mask = cv2.bitwise_and(mask, lm)

        # Morphological cleanup
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.dilate(mask, k, iterations=1)
        return mask

    def _redetect(self, bev: np.ndarray):
        """Add freshly detected features that are far from existing tracks."""
        new_pts = self._detect_features(bev)
        if new_pts is None:
            return

        existing = (self.tracked_pts.reshape(-1, 2)
                    if self.tracked_pts is not None and len(self.tracked_pts) > 0
                    else None)

        to_add = []
        for pt in new_pts.reshape(-1, 2):
            if existing is not None:
                dists = np.linalg.norm(existing - pt, axis=1)
                if dists.min() < 18:
                    continue
            to_add.append(pt)

        if not to_add:
            return

        new_arr  = np.array(to_add, np.float32).reshape(-1, 1, 2)
        new_ids  = [self._new_id() for _ in to_add]

        if self.tracked_pts is not None and len(self.tracked_pts) > 0:
            self.tracked_pts = np.vstack([self.tracked_pts, new_arr])
            self.pt_ids.extend(new_ids)
        else:
            self.tracked_pts = new_arr
            self.pt_ids      = new_ids

        # Cap total
        if len(self.pt_ids) > self.MAX_FEATURES:
            keep = self.MAX_FEATURES
            self.tracked_pts = self.tracked_pts[-keep:]
            self.pt_ids      = self.pt_ids[-keep:]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Video Processor (orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class VideoProcessor:
    """Orchestrates the full pipeline for a loaded video file."""

    CALIBRATE_EVERY     = 30    # frames
    MEASURE_INTERVAL    = 3.0   # seconds between table entries

    def __init__(self):
        self.cap:    Optional[cv2.VideoCapture] = None
        self.fps:    float = 30.0
        self.frame_count:  int   = 0
        self.is_processing: bool = False   # stays False until load_video() succeeds

        self.tf:      Optional[PerspectiveTransformer] = None
        self.lane_det: Optional[LaneLineDetector]      = None
        self.grid:    Optional[GridManager]            = None
        self.tracker: Optional[LaneMarkingTracker]     = None

        self.speed_table: List[Dict[str, Any]] = []
        self._last_measure_time: float = 0.0
        self._lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def load_video(self, path: str) -> bool:
        self.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            return False

        self.fps        = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.tf       = PerspectiveTransformer(fw, fh)
        self.lane_det = LaneLineDetector(fw, fh)
        self.grid     = GridManager(lane_width_m=3.5)
        self.tracker  = LaneMarkingTracker(self.grid)
        self.tracker.fps = self.fps

        self.frame_count        = 0
        self.speed_table        = []
        self._last_measure_time = 0.0
        self.is_processing      = True
        return True

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_processing = False

    # ── frame iteration ───────────────────────────────────────────────────

    def next_annotated_frame(self) -> Optional[np.ndarray]:
        """
        Read + process one frame; return the annotated BGR frame,
        or None when the video ends or the capture is not open.
        CV errors on a single frame are caught and the raw frame is returned
        so the MJPEG stream never crashes.
        """
        if not self.cap or not self.cap.isOpened():
            self.is_processing = False
            return None

        ret, frame = self.cap.read()
        if not ret:
            self.is_processing = False
            return None

        timestamp = self.frame_count / self.fps

        try:
            # ── 1. Auto-calibrate perspective from Hough lanes ───────────
            if self.frame_count % self.CALIBRATE_EVERY == 0:
                left, right = self.lane_det.detect(frame)
                if left and right:
                    self.tf.calibrate(left, right)

            # ── 2. Track lane markings in BEV ────────────────────────────
            self.tracker.update(frame, self.tf, timestamp)

            # ── 3. Draw grid overlay on camera frame ─────────────────────
            out = self.grid.draw_on_frame(frame.copy(), self.tf)

            # ── 4. Draw marking dots ─────────────────────────────────────
            out = self.tracker.draw_on_frame(out, self.tf)

            # ── 5. Record speed every MEASURE_INTERVAL seconds ───────────
            if timestamp - self._last_measure_time >= self.MEASURE_INTERVAL:
                speed = self.tracker.get_speed_kmh(
                    window_s=self.MEASURE_INTERVAL,
                    current_time=timestamp,
                )
                entry: Dict[str, Any] = {
                    "measurement": len(self.speed_table) + 1,
                    "time_start":  f"{self._last_measure_time:.1f}",
                    "time_end":    f"{timestamp:.1f}",
                    "speed_kmh":   round(speed, 1) if speed is not None else None,
                    "speed_ms":    round(speed / 3.6, 2) if speed is not None else None,
                    "confidence":  min(len(self.tracker.samples), 30),
                }
                with self._lock:
                    self.speed_table.append(entry)
                self._last_measure_time = timestamp

            # ── 6. HUD overlay ───────────────────────────────────────────
            speed_now = self.tracker.get_speed_kmh(3.0, timestamp)
            self._draw_hud(out, timestamp, speed_now)

        except Exception as exc:
            import traceback
            print(f"[processor] Frame {self.frame_count} error: {exc}")
            traceback.print_exc()
            # Fall back to the raw decoded frame so the stream stays alive
            out = frame.copy()
            cv2.putText(out, f"CV error: {exc}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.frame_count += 1
        return out

    # ── accessors ─────────────────────────────────────────────────────────

    def get_speed_table(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.speed_table)

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_processing": self.is_processing,
            "frame":         self.frame_count,
            "timestamp":     round(self.frame_count / self.fps, 2),
            "readings":      len(self.speed_table),
        }

    # ── HUD ───────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_hud(frame: np.ndarray, timestamp: float,
                  speed: Optional[float]):
        h, w = frame.shape[:2]

        # Speed box
        box_x2, box_y2 = min(w, 300), 65
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (box_x2, box_y2), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        if speed is not None:
            spd_txt = f"  {speed:.1f} km/h"
            cv2.putText(frame, spd_txt, (14, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1,
                        (0, 220, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "  -- km/h", (14, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1,
                        (120, 120, 120), 2, cv2.LINE_AA)

        # Timestamp
        cv2.putText(frame, f"t = {timestamp:.2f} s", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
