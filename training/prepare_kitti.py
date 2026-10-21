"""Download the KITTI 3D object dataset and cut out every usable vehicle.

    python training/prepare_kitti.py --data C:/Users/me/RoadSpeed-training

Needs ~13 GB free for the download (the images zip is 12 GB; it is read
directly, never unpacked).  The result is a few hundred MB:

    <data>/crops.bin     JPEG crops of the vehicles, back to back
    <data>/crops.npz     per crop: offset/length into crops.bin, size, orientation,
                         3D location, 2D box, camera, image id, class

KITTI (Geiger et al., CVPR 2012) is licensed CC BY-NC-SA 3.0: fine for study and
non-commercial projects; cite it if you publish results.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
FILES = {"labels": "data_object_label_2.zip", "calib": "data_object_calib.zip", "images": "data_object_image_2.zip"}
CLASSES = {"Car": 0, "Van": 1, "Truck": 2}
PRECROP = 192            # stored crop size (training cuts a jittered 128 px view out of it)
PRECROP_PAD = 0.35       # margin round the box in the stored crop


def download(url: str, dest: Path):
    """Download with resume and a progress line."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = have + int(r.headers.get("Content-Length", 0))
        if have and r.status != 206:          # server ignored the range: start again
            have, total = 0, int(r.headers.get("Content-Length", 0))
            tmp.unlink(missing_ok=True)
        t0, done = time.time(), have
        with open(tmp, "ab") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                el = max(time.time() - t0, 1e-3)
                rate = (done - have) / el
                eta = (total - done) / rate if rate > 0 else 0
                print(f"\r  {dest.name}: {done / 1e9:5.2f} / {total / 1e9:5.2f} GB  "
                      f"{rate / 1e6:5.1f} MB/s  ~{eta / 60:4.0f} min left ", end="", flush=True)
    print()
    tmp.replace(dest)


def ensure(data: Path, key: str) -> Path:
    p = data / FILES[key]
    if p.exists():
        try:
            with zipfile.ZipFile(p) as z:
                z.namelist()
            return p
        except zipfile.BadZipFile:
            print(f"  {p.name} is damaged - downloading again")
            p.unlink()
    print(f"Downloading {FILES[key]} ...")
    for attempt in range(20):                 # resume after dropped connections
        try:
            download(BASE + FILES[key], p)
            return p
        except Exception as e:                # noqa: BLE001 - network errors of all kinds
            print(f"\n  connection problem ({e}); retrying in 10 s ...")
            time.sleep(10)
    sys.exit(f"Could not download {FILES[key]}. Download it in a browser from {BASE + FILES[key]} "
             f"and put it in {data}, then run this again.")


def parse_calib(text: str):
    for line in text.splitlines():
        if line.startswith("P2:"):
            return np.array([float(v) for v in line.split()[1:13]]).reshape(3, 4)
    raise ValueError("no P2 in calib file")


def precrop(img, box):
    """Square crop of side max(w,h)*(1+2*PRECROP_PAD) round the box centre, PRECROP px,
    grey outside the image.  Returns the crop and (cx, cy, side) to map boxes into it."""
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + 2 * PRECROP_PAD)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = PRECROP / side
    a = np.array([[s, 0, PRECROP / 2 - cx * s], [0, s, PRECROP / 2 - cy * s]], np.float32)
    crop = cv2.warpAffine(img, a, (PRECROP, PRECROP), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    return crop, (cx, cy, side)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "RoadSpeed-training"))
    ap.add_argument("--min-height", type=float, default=18.0, help="smallest box height (px) to keep")
    a = ap.parse_args()
    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    out_bin, out_npz = data / "crops.bin", data / "crops.npz"
    if out_bin.exists() and out_npz.exists():
        print(f"Already prepared: {out_npz}  (delete it to prepare again)")
        return

    zl, zc, zi = (zipfile.ZipFile(ensure(data, k)) for k in ("labels", "calib", "images"))
    labels = sorted(n for n in zl.namelist() if n.startswith("training/label_2/") and n.endswith(".txt"))
    print(f"Cutting out vehicles from {len(labels)} images ...")
    rows = {k: [] for k in ("off", "len", "dims", "alpha", "loc", "box", "cam", "img", "cls", "trunc", "occ",
                            "pre")}
    off = 0
    with open(out_bin, "wb") as fb:
        for i, name in enumerate(labels):
            iid = Path(name).stem
            objs = []
            for line in zl.read(name).decode().splitlines():
                p = line.split()
                if not p or p[0] not in CLASSES:
                    continue
                trunc, occ, alpha = float(p[1]), int(p[2]), float(p[3])
                box = np.array([float(v) for v in p[4:8]])
                dims = np.array([float(v) for v in p[8:11]])          # h, w, l
                loc = np.array([float(v) for v in p[11:14]])
                if trunc > 0.4 or occ > 1 or box[3] - box[1] < a.min_height or loc[2] < 2:
                    continue
                objs.append((CLASSES[p[0]], trunc, occ, alpha, box, dims, loc))
            if not objs:
                continue
            P2 = parse_calib(zc.read(f"training/calib/{iid}.txt").decode())
            img = cv2.imdecode(np.frombuffer(zi.read(f"training/image_2/{iid}.png"), np.uint8), cv2.IMREAD_COLOR)
            f, cx, cy = P2[0, 0], P2[0, 2], P2[1, 2]
            # labels are in the reference camera; image_2 is offset by P2's translation
            shift = np.array([P2[0, 3] / f, P2[1, 3] / f, P2[2, 3]])
            for cls, trunc, occ, alpha, box, dims, loc in objs:
                crop, pre = precrop(img, box)
                ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                fb.write(jpg.tobytes())
                rows["off"].append(off)
                rows["len"].append(len(jpg))
                off += len(jpg)
                rows["dims"].append(dims)
                rows["alpha"].append(alpha)
                rows["loc"].append(loc + shift)
                rows["box"].append(box)
                rows["cam"].append([f, cx, cy, img.shape[1], img.shape[0]])
                rows["img"].append(int(iid))
                rows["cls"].append(cls)
                rows["trunc"].append(trunc)
                rows["occ"].append(occ)
                rows["pre"].append(pre)
            if (i + 1) % 250 == 0 or i + 1 == len(labels):
                print(f"\r  {i + 1}/{len(labels)} images, {len(rows['off'])} vehicles", end="", flush=True)
    print()
    np.savez(out_npz, **{k: np.array(v) for k, v in rows.items()})
    print(f"Saved {len(rows['off'])} vehicle crops to {data}")
    print(f"You can delete {FILES['images']} (12 GB) now if you need the space.")


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    main()
