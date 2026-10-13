"""Command-line use:  python cli.py input.mp4 [--out results_dir] [--hfov 100] [--height 1.3] ..."""
import argparse

from speedgrid.pipeline import Settings, process_video


def main():
    ap = argparse.ArgumentParser(description="Dashcam road grid + lane-marking speed estimation")
    ap.add_argument("video")
    ap.add_argument("--out", default="output")
    ap.add_argument("--hfov", type=float, default=100.0, help="camera horizontal FOV (deg)")
    ap.add_argument("--height", type=float, default=1.3, help="camera height above road (m)")
    ap.add_argument("--lane-width", type=float, default=3.5, help="known lane width for auto scale (0 = off)")
    ap.add_argument("--horizon", type=float, default=None, help="horizon row as fraction of height (default: auto)")
    ap.add_argument("--bonnet", type=float, default=0.0, help="fraction of frame bottom covered by bonnet")
    ap.add_argument("--grid-lines", type=int, default=12)
    ap.add_argument("--window", type=float, default=3.0)
    a = ap.parse_args()
    s = Settings(hfov_deg=a.hfov, cam_height_m=a.height, lane_width_m=a.lane_width or None,
                 horizon_frac=a.horizon, bonnet_frac=a.bonnet, grid_lines=a.grid_lines, window_s=a.window)
    res = process_video(a.video, a.out, s, progress=lambda p, m: None)
    print(f"{'win':>4} {'start':>6} {'end':>6} {'km/h':>7}  {'status':<18} measured from")
    for r in res["windows"]:
        print(f"{r['window']:>4} {r['start_s']:>6} {r['end_s']:>6} {str(r['speed_kmh']):>7}  "
              f"{r['status']:<18} {r.get('method', '')}")


if __name__ == "__main__":
    main()
