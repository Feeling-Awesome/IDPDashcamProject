"""Train the vehicle size + orientation network and export it for the app.

    python training/train_vehicle3d.py --data C:/Users/me/RoadSpeed-training

Input: the crops made by prepare_kitti.py.  Output (into speedgrid/models/):
    vehicle3d.onnx   the network (run by the app with OpenCV, no PyTorch needed)
    vehicle3d.json   input size, normalisation, size offsets and the measured accuracy

The network: ResNet-18 (ImageNet weights) with one small head predicting
    log(height, width, length) relative to the average vehicle, and
    (sin 2a, cos 2a) of the observation angle a (doubled: front and back of a box
    have the same outline, so the distance maths doesn't need to tell them apart).

The score that matters is how well these give the *distance*: at the end of
every few epochs the predicted size and angle are put through the same
geometry the app uses (speedgrid/box3d.py) on held-out KITTI images and the
distance error is reported.  The best epoch by that score is kept.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from speedgrid.box3d import box2d, crop_square, decode_angle, solve_location  # noqa: E402

PRECROP = 192
PRECROP_PAD = 0.35
INPUT = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Crops(torch.utils.data.Dataset):
    def __init__(self, data: Path, idx, meta, train: bool):
        self.path = data / "crops.bin"
        self.blob = None                  # opened lazily in each loader process (cheap to copy)
        self.idx, self.m, self.train = np.asarray(idx), meta, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        m = self.m
        if self.blob is None:
            self.blob = np.memmap(self.path, np.uint8, mode="r")
        jpg = np.asarray(self.blob[m["off"][i]: m["off"][i] + m["len"][i]])
        pre = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        # the box inside the stored crop
        cx, cy, side = m["pre"][i]
        s = PRECROP / side
        x1, y1, x2, y2 = m["box"][i]
        b = np.array([(x1 - cx) * s + PRECROP / 2, (y1 - cy) * s + PRECROP / 2,
                      (x2 - cx) * s + PRECROP / 2, (y2 - cy) * s + PRECROP / 2])
        alpha = float(m["alpha"][i])
        if self.train:
            # detector boxes are never exact: jitter size and position a little
            w, h = b[2] - b[0], b[3] - b[1]
            b = b + np.array([random.gauss(0, 0.05) * w, random.gauss(0, 0.05) * h,
                              random.gauss(0, 0.05) * w, random.gauss(0, 0.05) * h])
            img = crop_square(pre, b, INPUT)
            if random.random() < 0.5:                         # mirror image: angle mirrors too
                img = img[:, ::-1]
                alpha = float(wrap(np.pi - alpha))
            # dashcam-like quality: blur, low resolution, compression, colour shifts
            if random.random() < 0.3:
                img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.4, 1.4))
            if random.random() < 0.4:
                f = random.uniform(0.25, 0.7)
                small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                img = cv2.resize(small, (INPUT, INPUT), interpolation=cv2.INTER_LINEAR)
            if random.random() < 0.4:
                ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(25, 80)])
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            img = img.astype(np.float32)
            img = img * random.uniform(0.7, 1.3) + random.uniform(-25, 25)
            img = img * np.array([random.uniform(0.9, 1.1) for _ in range(3)], np.float32)
            img = np.clip(img, 0, 255)
        else:
            img = crop_square(pre, b, INPUT).astype(np.float32)
        x = (img[:, :, ::-1] / 255.0 - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()
        y_dims = torch.from_numpy(np.log(m["dims"][i]).astype(np.float32))
        y_ang = torch.tensor([math.sin(2 * alpha), math.cos(2 * alpha)], dtype=torch.float32)
        return x, y_dims, y_ang


class Net(nn.Module):
    def __init__(self, dims_mean):
        super().__init__()
        try:
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            self.body = torchvision.models.resnet18(weights=w)
        except Exception:                                      # older torchvision
            self.body = torchvision.models.resnet18(pretrained=True)
        self.body.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 5))
        self.register_buffer("dims_mean", torch.tensor(dims_mean, dtype=torch.float32))

    def forward(self, x):
        return self.body(x)          # [log dims - dims_mean (3), sin 2a, cos 2a]


def losses(out, y_dims, y_ang, dims_mean):
    l_dims = nn.functional.smooth_l1_loss(out[:, :3], y_dims - dims_mean, beta=0.05)
    a = nn.functional.normalize(out[:, 3:5], dim=1)
    l_ang = (1 - (a * y_ang).sum(1)).mean()
    return l_dims + 2.0 * l_ang, l_dims.item(), l_ang.item()


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    outs = []
    for x, _, _ in loader:
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            outs.append(model(x.to(dev, non_blocking=True)).float().cpu())
    return torch.cat(outs).numpy()


def evaluate(out, idx, m, dims_mean, n_depth=2500, seed=0):
    """Size error, angle error and - what matters - distance error through the geometry."""
    dims_p = np.exp(out[:, :3] + dims_mean)
    alpha_p = decode_angle(out[:, 3:5])
    dims_t = m["dims"][idx]
    alpha_t = m["alpha"][idx]
    size_err = np.abs(dims_p / dims_t - 1)
    d_ang = np.abs(wrap(2 * (alpha_p - alpha_t))) / 2
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(idx), size=min(n_depth, len(idx)), replace=False)
    rel = []
    for j in pick:
        i = idx[j]
        f, cx, cy, W, H = m["cam"][i]
        cam = (f, cx, cy, np.eye(3))
        b = m["box"][i]
        valid = (b[0] > 1, b[1] > 1, b[2] < W - 2, b[3] < H - 2)
        loc, _ = solve_location(b, dims_p[j], alpha_p[j], cam, valid)
        if loc is not None:
            rel.append(math.log(loc[2] / m["loc"][i][2]))
    rel = np.array(rel)
    return {
        "height_err_pct": round(float(np.median(size_err[:, 0])) * 100, 2),
        "width_err_pct": round(float(np.median(size_err[:, 1])) * 100, 2),
        "length_err_pct": round(float(np.median(size_err[:, 2])) * 100, 2),
        "angle_err_deg_median": round(float(np.degrees(np.median(d_ang))), 2),
        "distance_err_pct_median": round(float(np.median(np.abs(rel))) * 100, 2),
        "distance_err_pct_90": round(float(np.percentile(np.abs(rel), 90)) * 100, 2),
        "distance_within_10pct": round(float(np.mean(np.abs(rel) < math.log(1.1))) * 100, 1),
        "depth_rel_sd": round(float(1.4826 * np.median(np.abs(rel - np.median(rel)))), 4),
        "distance_bias_pct": round(float(np.median(rel)) * 100, 2),
        "n_distance": int(len(rel)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "RoadSpeed-training"))
    ap.add_argument("--out", default=str(ROOT / "speedgrid" / "models"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    data, out_dir = Path(a.data), Path(a.out)
    m = dict(np.load(data / "crops.npz"))
    n = len(m["off"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{n} vehicle crops | device: {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU (slow!)'}")

    # split by image so the same scene never appears in both
    imgs = np.unique(m["img"])
    rng = np.random.default_rng(42)
    val_imgs = set(rng.choice(imgs, size=int(0.15 * len(imgs)), replace=False).tolist())
    is_val = np.array([i in val_imgs for i in m["img"]])
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    dims_mean = np.log(m["dims"][tr_idx]).mean(0).astype(np.float32)
    print(f"train {len(tr_idx)} | validation {len(va_idx)} | average size (h, w, l) "
          f"{np.round(np.exp(dims_mean), 2).tolist()} m")

    workers = a.workers if sys.platform != "win32" else min(a.workers, 4)
    tr = torch.utils.data.DataLoader(Crops(data, tr_idx, m, True), batch_size=a.batch, shuffle=True,
                                     num_workers=workers, pin_memory=dev.type == "cuda", drop_last=True,
                                     persistent_workers=workers > 0)
    va = torch.utils.data.DataLoader(Crops(data, va_idx, m, False), batch_size=256, shuffle=False,
                                     num_workers=workers, pin_memory=dev.type == "cuda")
    model = Net(dims_mean).to(dev)
    head = list(model.body.fc.parameters())
    body = [p for n_, p in model.body.named_parameters() if not n_.startswith("fc.")]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr * 0.3}, {"params": head, "lr": a.lr}], weight_decay=1e-4)
    steps = a.epochs * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr * 0.3, a.lr], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    dm = torch.tensor(dims_mean, device=dev)

    best, best_state, best_metrics = 1e9, None, None
    t_start = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, nd, na, nb = time.time(), 0.0, 0.0, 0.0, 0
        for x, yd, ya in tr:
            x, yd, ya = x.to(dev, non_blocking=True), yd.to(dev), ya.to(dev)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                out = model(x)
            loss, ld, la = losses(out.float(), yd, ya, dm)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, nd, na, nb = tot + loss.item(), nd + ld, na + la, nb + 1
        msg = f"epoch {ep:2d}/{a.epochs}  loss {tot / nb:.4f} (size {nd / nb:.4f}, angle {na / nb:.4f})  {time.time() - t0:.0f}s"
        if ep % 3 == 0 or ep == a.epochs:
            out = predict(model, va, dev)
            met = evaluate(out, va_idx, m, dims_mean)
            msg += (f" | val: height {met['height_err_pct']}%  angle {met['angle_err_deg_median']} deg  "
                    f"DISTANCE {met['distance_err_pct_median']}% (90%: {met['distance_err_pct_90']}%)")
            score = met["distance_err_pct_median"] + 0.3 * met["distance_err_pct_90"]
            if score < best:
                best, best_metrics = score, met
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                msg += "  <- best"
        print(msg, flush=True)

    model.load_state_dict(best_state)
    model.eval().cpu()
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "vehicle3d.onnx"
    dummy = torch.zeros(1, 3, INPUT, INPUT)
    kw = dict(opset_version=12, input_names=["image"], output_names=["out"],
              dynamic_axes={"image": {0: "n"}, "out": {0: "n"}})
    try:                      # newer PyTorch defaults to a different exporter: ask for the classic one
        torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, str(onnx_path), **kw)
    meta = {"input_size": INPUT, "mean_rgb": MEAN, "std_rgb": STD, "dims_mean": dims_mean.tolist(),
            "depth_rel_sd": max(best_metrics["depth_rel_sd"], 0.05), "validation": best_metrics,
            "trained": time.strftime("%Y-%m-%d %H:%M"), "epochs": a.epochs, "train_crops": int(len(tr_idx)),
            "data": "KITTI 3D object (CC BY-NC-SA 3.0)", "minutes": round((time.time() - t_start) / 60, 1)}
    (out_dir / "vehicle3d.json").write_text(json.dumps(meta, indent=2))

    # check the exported file gives the same answers through OpenCV (what the app uses)
    net = cv2.dnn.readNet(str(onnx_path))
    x = torch.randn(4, 3, INPUT, INPUT)
    net.setInput(x.numpy())
    diff = float(np.abs(net.forward() - model(x).detach().numpy()).max())
    print(f"\nSaved {onnx_path} (+ vehicle3d.json). OpenCV check: max difference {diff:.2e}")
    print("Validation on held-out KITTI images (distance from size + angle, same maths as the app):")
    for k, v in best_metrics.items():
        print(f"  {k}: {v}")
    if diff > 1e-2:
        print("WARNING: the OpenCV result differs from PyTorch - tell Claude.")


if __name__ == "__main__":
    main()
