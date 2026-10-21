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


def load_sprites(paths):
    """Car photos (rear views) used as other traffic; physical width assumed 1.85 m."""
    out = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise SystemExit(f"cannot read sprite {p}")
        out.append(img.astype(np.float32))
    return out


# other traffic for --traffic: lane offset (m), absolute speed (km/h), start distance ahead (m), sprite
TRAFFIC = [
    {"id": 1, "d": 3.5, "kmh": 100.0, "z0": 9.0, "sprite": 1},     # right lane, faster than us
    {"id": 2, "d": -3.5, "kmh": 70.0, "z0": 55.0, "sprite": 0},    # left lane, we overtake it
    {"id": 3, "d": 0.0, "kmh": None, "z0": 24.0, "sprite": 0},     # our lane: same speed as us + 8
]


def draw_sprites(img, cam, cars, sprites):
    """Paste car photos standing on the road, far ones first.  Returns their true boxes."""
    H, W = img.shape[:2]
    boxes = []
    for c in sorted(cars, key=lambda c: -c["z"]):
        z = c["z"]
        if z < 2.5 or z > 150:
            continue
        sp = sprites[c["sprite"]]
        p = cam.project_3d(np.array([c["d"] - 0.925, c["d"] + 0.925]), np.zeros(2), np.array([z, z]))
        w = p[1, 0] - p[0, 0]
        if not np.all(np.isfinite(p)) or w < 4:
            continue
        h = w * sp.shape[0] / sp.shape[1]
        x0, y1 = p[0, 0], (p[0, 1] + p[1, 1]) / 2 + 2 * w / sp.shape[1]     # sprite has a 2 px margin below
        xi0, yi0 = int(round(x0)), int(round(y1 - h))
        wi, hi = max(int(round(w)), 2), max(int(round(h)), 2)
        spr = cv2.resize(sp, (wi, hi), interpolation=cv2.INTER_AREA)
        ax0, ay0, ax1, ay1 = max(xi0, 0), max(yi0, 0), min(xi0 + wi, W), min(yi0 + hi, H)
        if ax1 <= ax0 or ay1 <= ay0:
            continue
        img[ay0:ay1, ax0:ax1] = spr[ay0 - yi0:ay1 - yi0, ax0 - xi0:ax1 - xi0]
        boxes.append({"id": c["id"], "z": round(z, 3), "speed_kmh": round(c["v"] * 3.6, 3),
                      "box": [ax0, ay0, ax1, ay1]})
    return boxes


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
    ap.add_argument("--scene", choices=["highway", "outback", "bare"], default="highway",
                    help="highway: 3 lanes with dashes; outback: 2-lane road with only solid edge "
                         "lines, worn wheel tracks, sand verges; bare: no markings at all")
    ap.add_argument("--twotone", type=float, default=6.0, help="how much darker the other lane is")
    ap.add_argument("--patchiness", type=float, default=2.5, help="road patches/cracks density")
    ap.add_argument("--bounce", type=float, default=0.0, help="pitch bounce amplitude (deg)")
    ap.add_argument("--traffic", nargs="*", default=None,
                    help="car sprite images: add other cars with known speeds (ground truth in .truth.json)")
    ap.add_argument("--degrade", action="store_true",
                    help="motion blur + 854x480 heavy compression (like a re-uploaded web video)")
    args = ap.parse_args()

    W, H = args.w, args.h
    vp = (W / 2 + 15, 0.46 * H)
    cam = CameraModel(W, H, hfov_deg=args.hfov, cam_height_m=args.height, vp=vp)
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)

    def geometry(cam):
        inv = np.linalg.inv(cam.G2I)
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
        return ground, Z, d, s_loc

    ground, Z, d, s_loc = geometry(cam)

    rng = np.random.default_rng(1)
    tex_asph = cv2.GaussianBlur(rng.normal(0, 1, (512, 512)).astype(np.float32), (0, 0), 1.2)
    tex_grass = cv2.GaussianBlur(rng.normal(0, 1, (512, 512)).astype(np.float32), (0, 0), 2.0)
    tex_veg = tex_grass / tex_grass.std()            # unit-variance copy for verge texture
    # realistic worn asphalt: faint grain + sparse patches, cracks and tar seams
    patches = np.zeros((2048, 512), np.float32)          # 20 px/m -> 102 m x 25.6 m tile
    for _ in range(int(140 * args.patchiness)):
        cx, cy = rng.integers(0, 512), rng.integers(0, 2048)
        ax, ay = rng.integers(4, 30), rng.integers(4, 40)
        cv2.ellipse(patches, (int(cx), int(cy)), (int(ax), int(ay)), float(rng.uniform(0, 180)), 0, 360,
                    float(rng.choice([-1, 1]) * rng.uniform(4, 12)), -1)
    for _ in range(int(90 * args.patchiness)):
        x0, y0 = rng.integers(0, 512), rng.integers(0, 2048)
        pts = np.cumsum(rng.normal(0, 6, (8, 2)), axis=0) + [x0, y0]
        cv2.polylines(patches, [pts.astype(np.int32)], False, -float(rng.uniform(8, 20)), 1)
    patches = cv2.GaussianBlur(patches, (0, 0), 1.0)
    sky = np.linspace(200, 235, H)[:, None].repeat(W, 1)

    # pixel footprint (m) for crude anti-aliasing of the paint
    fp = np.clip(Z / cam.focal, 0.002, 1.0)

    def paint(dist, half):
        return np.clip((half - np.abs(dist)) / fp + 0.5, 0, 1)

    dt = 1 / args.fps
    n = int(args.seconds * args.fps)
    tmp = Path(args.out).with_suffix(".raw.mp4")
    enc = (["-vf", "scale=854:480", "-crf", "30"] if args.degrade else ["-crf", "20"])
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
                             *enc, "-g", "30", "-pix_fmt", "yuv420p", args.out], stdin=subprocess.PIPE)
    s_car = 0.0
    truth = []
    sprites = load_sprites(args.traffic) if args.traffic else None
    others = [dict(c, s=c["z0"]) for c in TRAFFIC] if sprites else []
    for i in range(n):
        t = i * dt
        if args.stopgo:
            v = stopgo_speed(t, args.v0) / 3.6
        else:
            v = (args.v0 + (args.v1 - args.v0) * t / args.seconds) / 3.6
        if args.bounce:
            cam_t = CameraModel(W, H, hfov_deg=args.hfov, cam_height_m=args.height,
                                vp=(vp[0], vp[1] + cam.focal * np.tan(np.radians(
                                    args.bounce * np.sin(2 * np.pi * 1.7 * t)))))
            ground, Z, d, s_loc = geometry(cam_t)
        S = s_car + s_loc
        si = (S * 20).astype(np.int64) % 512
        di = ((d + 50) * 20).astype(np.int64) % 512
        a = tex_asph[si, di]
        g = tex_grass[(S * 8).astype(np.int64) % 512, ((d + 50) * 8).astype(np.int64) % 512]

        img = np.zeros((H, W, 3), np.float32)
        img[..., 0] = sky * 1.0
        img[..., 1] = sky * 0.93
        img[..., 2] = sky * 0.85
        if args.scene == "highway":
            on_road = np.abs(d) < 6.0
            asph = 95 + 8 * a
            grass = np.stack([45 + 10 * g, 120 + 18 * g, 70 + 10 * g], -1)
            m = ground & on_road
            img[m] = np.stack([asph, asph, asph * 1.02], -1)[m]
            m2 = ground & ~on_road
            img[m2] = grass[m2]
            w_edge = np.maximum(paint(d - 5.25, 0.075), paint(d + 5.25, 0.075))
            phase = np.mod(S, 9.0)
            dash_on = np.clip(np.minimum(phase, 3.0 - phase) / fp + 0.5, 0, 1)
            w_dash = np.maximum(paint(d - 1.75, 0.075), paint(d + 1.75, 0.075)) * dash_on
            wpaint = np.maximum(w_edge, w_dash) * ground
        else:
            # 2-lane rural road, car in the left lane; worn lighter wheel tracks
            left, right = -1.95, 5.25
            on_road = (d > left) & (d < right)
            pa = patches[(S * 20).astype(np.int64) % 2048, ((d + 50) * 20).astype(np.int64) % 512]
            track = np.exp(-((np.abs(d) - 0.0) / 1.1) ** 2) * 18      # lighter worn band in our lane
            asph = 92 + 5 * a + pa + track - args.twotone * (d > 1.75)
            # reddish sand with gravel and dense low scrub (BGR)
            g1 = tex_veg[(S * 8).astype(np.int64) % 512, ((d + 50) * 8).astype(np.int64) % 512]
            g2 = tex_veg[(S * 4).astype(np.int64) % 512, ((d + 50) * 4).astype(np.int64) % 512]
            sand = np.stack([95 + 6 * g1 + 6 * g2, 125 + 7 * g1 + 7 * g2, 175 + 8 * g1 + 8 * g2], -1)
            bush = tex_veg[(S * 1.6).astype(np.int64) % 512, ((d + 50) * 1.6).astype(np.int64) % 512] > 1.0
            bush &= np.abs(d - 1.6) > 5.0
            sand[bush] = np.stack([55 + 8 * g1, 85 + 10 * g1, 80 + 8 * g1], -1)[bush]
            m = ground & on_road
            img[m] = np.stack([asph, asph * 1.01, asph * 1.03], -1)[m]
            m2 = ground & ~on_road
            img[m2] = sand[m2]
            if args.scene == "outback":
                wpaint = np.maximum(paint(d - (right - 0.25), 0.05), paint(d - (left + 0.25), 0.05)) * ground
            else:
                wpaint = np.zeros_like(d)
        img = img * (1 - wpaint[..., None]) + 225 * wpaint[..., None]

        if args.stopgo:
            draw_traffic(img, cam, t, s_car, v)
        vboxes = []
        if sprites:
            for c in others:
                c["v"] = c["kmh"] / 3.6 if c["kmh"] is not None else v + 8 / 3.6
                c["z"] = c["s"]                       # straight road: distance ahead = gap
            vboxes = draw_sprites(img, cam, others, sprites)
        img += rng.normal(0, 3, img.shape).astype(np.float32)
        frame = np.clip(img, 0, 255).astype(np.uint8)
        if args.degrade:   # motion blur (vertical, grows with speed) + soft optics
            kl = int(1 + v / 8) | 1
            frame = cv2.blur(frame, (1, kl))
            frame = cv2.GaussianBlur(frame, (0, 0), 0.9)
        proc.stdin.write(frame.tobytes())
        truth.append({"t": round(t, 4), "speed_kmh": round(v * 3.6, 3), **({"vehicles": vboxes} if sprites else {})})
        s_car += v * dt
        for c in others:
            c["s"] += (c["v"] - v) * dt
    proc.stdin.close()
    proc.wait()
    Path(args.out).with_suffix(".truth.json").write_text(json.dumps(
        {"hfov": args.hfov, "height": args.height, "vp": vp, "truth": truth}))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
