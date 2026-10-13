"""Lane-marking detection in the bird's-eye view.

Lane paint is a thin, bright (white) or saturated yellow stripe.  In the BEV
it runs roughly along the image columns, so a *horizontal white top-hat*
(original minus morphological opening with a kernel wider than the paint)
isolates it while suppressing broad bright areas (sky reflections, sunlit
patches).  The top-hat response is thresholded adaptively against the
statistics of the road surface, combined with an HSV yellow mask, cleaned
morphologically and split into connected components.  Each elongated
component is a marking; the endpoints of *dashed* segments are the features
that get tracked (a solid line has no longitudinal feature to track).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import CameraModel


@dataclass
class MarkingEnd:
    x: float          # BEV px
    y: float
    kind: str         # "near" (end closest to the car) or "far"
    comp: int         # component id in this frame


def detect_markings(bev_bgr: np.ndarray, cam: CameraModel, road_mask: np.ndarray | None,
                    stats_mask: np.ndarray | None = None, return_tophat: bool = False):
    ppm = cam.ppm
    Hb, Wb = bev_bgr.shape[:2]
    valid = cam.valid_bev_mask()

    L = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.7 * ppm) | 1, 1))
    tophat = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k)

    region = valid
    if road_mask is not None:
        grow = cv2.getStructuringElement(cv2.MORPH_RECT, (int(1.0 * ppm) | 1, 3))
        region = cv2.bitwise_and(valid, cv2.dilate(road_mask, grow))
    # threshold statistics from the road surface itself (a textured verge inside a wide
    # search area would otherwise raise the threshold and hide thin edge lines)
    stat_region = region if stats_mask is None else cv2.bitwise_and(valid, stats_mask)
    hist = cv2.calcHist([tophat], [0], stat_region, [256], [0, 256]).ravel()
    if hist.sum() < 100:
        empty = np.zeros((Hb, Wb), np.uint8)
        return (empty, [], None) if return_tophat else (empty, [])
    # median / MAD from the histogram (tophat is uint8): no sorting needed
    med = _hist_median(hist)
    dev = np.abs(np.arange(256) - med).astype(np.int64)
    mad = _hist_median(np.bincount(dev, weights=hist, minlength=256)) * 1.4826 + 1e-3
    thr = max(18.0, med + 6.0 * mad)
    _, white = cv2.threshold(tophat, thr, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (12, 70, 110), (38, 255, 255))
    _, ysat = cv2.threshold(cv2.morphologyEx(hsv[:, :, 1], cv2.MORPH_TOPHAT, k), 40, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(cv2.bitwise_or(white, cv2.bitwise_and(yellow, ysat)), region)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(0.25 * ppm) | 1)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, int(0.45 * ppm) | 1)))

    min_len = 0.5 * ppm
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    accept = np.zeros(n, np.uint8)
    ends: list[MarkingEnd] = []
    inner_valid = _inner_valid(cam)
    min_area = 0.15 * ppm * ppm * 0.4
    cand = np.flatnonzero((stats[:, cv2.CC_STAT_AREA] >= min_area) &
                          (np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= min_len))
    for i in cand:
        if i == 0:
            continue
        x, y, w, h, area = stats[i]
        ys, xs = np.nonzero(lbl[y:y + h, x:x + w] == i)
        ys = ys + y
        xs = xs + x
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        long_side, short_side = max(rw, rh), min(rw, rh)
        if long_side < min_len or short_side > 0.6 * ppm or long_side < 2.5 * short_side:
            continue
        accept[i] = 255
        if long_side > 7.0 * ppm:
            continue        # a solid line: its "ends" are just where detection fades - not real
        # Endpoints along the road direction (BEV y).  Clipped ends are useless.
        for kind, yy in (("far", ys.min()), ("near", ys.max())):
            row_x = xs[ys == yy]
            ex = float(row_x.mean())
            # refine with the mean of a 2-px band for sub-pixel stability
            band = np.abs(ys - yy) <= 1
            ey = float(ys[band].mean())
            xi, yi = int(round(ex)), int(round(yy))
            step = -6 if kind == "far" else 6
            yb = yi + step
            if not (0 <= yb < Hb) or not inner_valid[yb, min(max(xi, 0), Wb - 1)]:
                continue
            if yi <= 3 or yi >= Hb - 4:
                continue
            ends.append(MarkingEnd(ex, ey, kind, i))
    if return_tophat:
        return accept[lbl], ends, tophat
    return accept[lbl], ends


def _hist_median(hist) -> float:
    c = np.cumsum(hist)
    return float(np.searchsorted(c, c[-1] / 2.0))


def _inner_valid(cam):
    v = getattr(cam, "_inner_valid", None)
    if v is None:
        v = cv2.erode(cam.valid_bev_mask(), np.ones((9, 9), np.uint8)) > 0
        cam._inner_valid = v
    return v


def measure_lane_spacing(mask: np.ndarray, cam: CameraModel) -> list[float]:
    """Lateral distances (m) between adjacent marking lines, sampled per 1 m band."""
    ppm = cam.ppm
    Hb = mask.shape[0]
    res = []
    for y0 in range(int(Hb - 18 * ppm), int(Hb - 3 * ppm), int(ppm)):
        if y0 < 0:
            continue
        band = mask[y0:y0 + int(ppm)].max(axis=0) > 0
        idx = np.flatnonzero(band)
        if len(idx) < 2:
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1)
        centres = sorted(float(r.mean()) for r in runs if len(r) <= 0.6 * ppm)
        for a, b in zip(centres, centres[1:]):
            d = (b - a) / ppm
            if 2.2 < d < 5.5:
                res.append(d)
    return res
