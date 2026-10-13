"""Render a synthetic dashcam video with a *known* speed profile (ground truth).

The road is ray-cast through the same pinhole/flat-ground camera model:
3 lanes x 3.5 m, solid edge lines, 3 m dashes with 6 m gaps, grass verges,
optional constant curvature.  Speed ramps linearly so the per-window
ground truth changes over time.

    python tools/make_synthetic.py out.mp4 --v0 60 --v1 90 --seconds 15 --curv 0.003
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speedgrid.camera import CameraModel  # noqa: E402


def stopgo_speed(t, v0):
    """0-6 s cruise, 6-10 s brake, 10-22 s stopped, 22-27 s accelerate, then cruise (km/h)."""
    if t < 6:
        return v0
    if t < 10:
        return v0 * (10 - t) / 4
    if t < 22:
        return 0.0
    if t < 27:
        return v0 * (t - 22) / 5
    return v0


def _box(img, cam, d0, d1, z0, z1, h, colour):
    """Draw a solid box (vehicle) standing on the road; painter's order inside the box."""
    faces = []
    zc0 = max(z0, 0.4)
    if z1 <= 0.4:
        return
    if d0 > 0:      # box right of camera: its left side is visible
        faces.append([(d0, 0, zc0), (d0, 0, z1), (d0, h, z1), (d0, h, zc0)])
    elif d1 < 0:
        faces.append([(d1, 0, zc0), (d1, 0, z1), (d1, h, z1), (d1, h, zc0)])
    if h < cam.cam_height_m:
        faces.append([(d0, h, zc0), (d1, h, zc0), (d1, h, z1), (d0, h, z1)])
    if z0 >= 0.4:
        faces.append([(d0, 0, z0), (d1, 0, z0), (d1, h, z0), (d0, h, z0)])
    for k, f in enumerate(faces):
        f = np.array(f, float)
        p = cam.project_3d(f[:, 0], f[:, 1], f[:, 2])
        shade = 0.8 if k == 0 and len(faces) > 1 else 1.0
        cv2.fillPoly(img, [np.round(p).astype(np.int32)], tuple(c * shade for c in colour), cv2.LINE_AA)


def draw_traffic(img, cam, t, s_car, v):
    # lead box-truck in our lane, keeps a speed-dependent gap
    gap = 6.5 + 1.1 * v
    _box(img, cam, -1.25, 1.25, gap, gap + 7, 3.4, (235, 235, 235))
    # big dark truck in the right lane, a bit slower than us (stops when we stop)
    rel = 4.0 - 0.12 * s_car
    rel = (rel + 30) % 60 - 30
    _box(img, cam, 2.3, 4.8, rel, rel + 13, 3.8, (70, 55, 45))
    # car in the left lane creeping forward at 12 km/h even while we are stopped
    rel_l = (3.0 + 12 / 3.6 * t - s_car + 10) % 50 - 10
    _box(img, cam, -4.6, -2.7, rel_l, rel_l + 4.5, 1.5, (40, 40, 160))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--v0", type=float, default=60, help="start speed km/h")
    ap.add_argument("--v1", type=float, default=90, help="end speed km/h")
    ap.add_argument("--curv", type=float, default=0.0, help="road curvature 1/m (+ = right)")
    ap.add_argument("--hfov", type=float, default=100)
    ap.add_argument("--height", type=float, default=1.35)
    ap.add_argument("--stopgo", action="store_true",
                    help="cruise at v0, brake to a full stop, wait, pull away (with traffic)")
    args = ap.parse_args()

    W, H = args.w, args.h
    vp = (W / 2 + 15, 0.46 * H)
    cam = CameraModel(W, H, hfov_deg=args.hfov, cam_height_m=args.height, vp=vp)
    inv = np.linalg.inv(cam.G2I)
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    P = np.stack([uu, vv, np.ones_like(uu)], -1) @ inv.T
    ground = P[..., 2] > 1e-9
    X = P[..., 0] / np.where(ground, P[..., 2], 1)
    Z = P[..., 1] / np.where(ground, P[..., 2], 1)
    ground &= (Z > 0) & (Z < 400)

    k = args.curv
    if abs(k) < 1e-9:
        d, s_loc = X, Z
    else:
        R = 1 / k
        r = np.sqrt((X - R) ** 2 + Z ** 2)
        d = np.sign(R) * (abs(R) - r)
        s_loc = abs(R) * np.arctan2(Z, np.sign(R) * (R - X))

    rng = np.random.default_rng(1)
    tex_asph = cv2.GaussianBlur(rng.normal(0, 1, (512, 512)).astype(np.float32), (0, 0), 1.2)
    tex_grass = cv2.GaussianBlur(rng.normal(0, 1, (512, 512)).astype(np.float32), (0, 0), 2.0)
    sky = np.linspace(200, 235, H)[:, None].repeat(W, 1)

    # pixel footprint (m) for crude anti-aliasing of the paint
    fp = np.clip(Z / cam.focal, 0.002, 1.0)

    def paint(dist, half):
        return np.clip((half - np.abs(dist)) / fp + 0.5, 0, 1)

    dt = 1 / args.fps
    n = int(args.seconds * args.fps)
    tmp = Path(args.out).with_suffix(".raw.mp4")
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
                             "-crf", "20", "-pix_fmt", "yuv420p", args.out], stdin=subprocess.PIPE)
    s_car = 0.0
    truth = []
    for i in range(n):
        t = i * dt
        if args.stopgo:
            v = stopgo_speed(t, args.v0) / 3.6
        else:
            v = (args.v0 + (args.v1 - args.v0) * t / args.seconds) / 3.6
        S = s_car + s_loc
        si = (S * 20).astype(np.int64) % 512
        di = ((d + 50) * 20).astype(np.int64) % 512
        a = tex_asph[si, di]
        g = tex_grass[(S * 8).astype(np.int64) % 512, ((d + 50) * 8).astype(np.int64) % 512]

        img = np.zeros((H, W, 3), np.float32)
        img[..., 0] = sky * 1.0
        img[..., 1] = sky * 0.93
        img[..., 2] = sky * 0.85
        on_road = np.abs(d) < 6.0
        asph = 95 + 8 * a
        grass = np.stack([45 + 10 * g, 120 + 18 * g, 70 + 10 * g], -1)
        m = ground & on_road
        img[m] = np.stack([asph, asph, asph * 1.02], -1)[m]
        m2 = ground & ~on_road
        img[m2] = grass[m2]

        # markings
        w_edge = np.maximum(paint(d - 5.25, 0.075), paint(d + 5.25, 0.075))
        phase = np.mod(S, 9.0)
        dash_on = np.clip(np.minimum(phase, 3.0 - phase) / fp + 0.5, 0, 1)
        w_dash = np.maximum(paint(d - 1.75, 0.075), paint(d + 1.75, 0.075)) * dash_on
        wpaint = np.maximum(w_edge, w_dash) * ground
        img = img * (1 - wpaint[..., None]) + 225 * wpaint[..., None]

        if args.stopgo:
            draw_traffic(img, cam, t, s_car, v)
        img += rng.normal(0, 3, img.shape).astype(np.float32)
        frame = np.clip(img, 0, 255).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
        truth.append({"t": round(t, 4), "speed_kmh": round(v * 3.6, 3)})
        s_car += v * dt
    proc.stdin.close()
    proc.wait()
    Path(args.out).with_suffix(".truth.json").write_text(json.dumps(
        {"hfov": args.hfov, "height": args.height, "vp": vp, "truth": truth}))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
