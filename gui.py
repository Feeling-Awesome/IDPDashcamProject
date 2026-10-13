"""Road Speed Grid - desktop app.

Open a dashcam video, watch it being analysed, then play back the annotated
video next to the speed table.  Run with:  python gui.py   (or double-click the
built app / RoadSpeed.pyw).  A video path given on the command line (e.g. a
file dropped onto the app icon) is opened straight away.
"""
from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speedgrid.pipeline import Settings, process_video  # noqa: E402

APP_NAME = "Road Speed Grid"
VIDEO_TYPES = [("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.webm *.MP4 *.MOV *.AVI"), ("All files", "*.*")]


def results_root() -> Path:
    docs = Path.home() / "Documents"
    root = (docs if docs.is_dir() else Path.home()) / "RoadSpeed"
    root.mkdir(parents=True, exist_ok=True)
    return root


def open_in_file_manager(path: Path):
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def bgr_to_photo(img: np.ndarray, max_w: int, max_h: int) -> tk.PhotoImage:
    """Fit a BGR frame into max_w x max_h and wrap it as a Tk image (PPM, no Pillow needed)."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    if scale <= 0:
        scale = 1.0
    nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = f"P6 {nw} {nh} 255 ".encode() + rgb.tobytes()
    return tk.PhotoImage(width=nw, height=nh, data=data, format="PPM")


class SettingsPanel(ttk.Frame):
    FIELDS = [
        ("hfov_deg", "Camera field of view (°)", 100.0),
        ("cam_height_m", "Camera height (m)", 1.3),
        ("lane_width_m", "Lane width for auto-scale (m, 0 = off)", 3.5),
        ("horizon_frac", "Horizon (% from top, blank = auto)", None),
        ("bonnet_frac", "Bonnet at bottom (%, 0 = auto)", 0.0),
        ("grid_lines", "Grid lines (1 m apart)", 12),
        ("window_s", "Report every (seconds)", 3.0),
        ("speedup", "Video sped up × (1 = real time)", 1.0),
    ]

    def __init__(self, master):
        super().__init__(master, padding=(12, 6, 12, 10))
        self.vars: dict[str, tk.StringVar] = {}
        for i, (key, label, default) in enumerate(self.FIELDS):
            r, c = divmod(i, 4)
            box = ttk.Frame(self)
            box.grid(row=r, column=c, sticky="w", padx=(0, 18), pady=4)
            ttk.Label(box, text=label, foreground="#666").pack(anchor="w")
            v = tk.StringVar(value="" if default is None else str(default))
            ttk.Entry(box, textvariable=v, width=12).pack(anchor="w")
            self.vars[key] = v

    # remembered between sessions (a dashcam's FOV / mounting rarely changes)

    @classmethod
    def store_path(cls) -> Path:
        return results_root() / "settings.json"

    def load(self):
        try:
            data = json.loads(self.store_path().read_text())
            for k, v in data.items():
                if k in self.vars:
                    self.vars[k].set(v)
        except (OSError, ValueError):
            pass

    def save(self):
        try:
            self.store_path().write_text(json.dumps({k: v.get() for k, v in self.vars.items()}, indent=1))
        except OSError:
            pass

    def settings(self) -> Settings:
        def num(key, default, lo, hi):
            try:
                return min(max(float(self.vars[key].get()), lo), hi)
            except ValueError:
                return default

        horizon = self.vars["horizon_frac"].get().strip()
        lane = num("lane_width_m", 3.5, 0, 6)
        try:
            hz = min(max(float(horizon), 10), 90) / 100.0 if horizon else None
        except ValueError:
            hz = None
        return Settings(
            hfov_deg=num("hfov_deg", 100.0, 30, 170),
            cam_height_m=num("cam_height_m", 1.3, 0.5, 4.0),
            lane_width_m=lane if lane > 0 else None,
            horizon_frac=hz,
            bonnet_frac=num("bonnet_frac", 0.0, 0, 50) / 100.0,
            grid_lines=int(num("grid_lines", 12, 3, 40)),
            window_s=num("window_s", 3.0, 0.5, 60),
            speedup=num("speedup", 1.0, 0.1, 100),
        )


class App:
    def __init__(self, root: tk.Tk, initial: str | None = None):
        self.root = root
        root.title(APP_NAME)
        root.minsize(980, 600)
        self.video_path: Path | None = None
        self.out_dir: Path | None = None
        self.result: dict | None = None
        self.worker: threading.Thread | None = None
        self.cancel = threading.Event()
        self.msgs: queue.Queue = queue.Queue()
        self.preview_lock = threading.Lock()
        self.preview_frame = None
        self.photo = None
        # player
        self.cap = None
        self.n_frames = 0
        self.fps = 30.0
        self.cur = 0
        self.playing = False
        self.last_frame = None
        self._seeking = False

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(40, self._poll)
        if initial:
            self.set_video(Path(initial))

    # ------------------------------------------------------------------ layout
    def _build(self):
        top = ttk.Frame(self.root, padding=(12, 10, 12, 4))
        top.pack(fill="x")
        ttk.Button(top, text="Open video…", command=self.choose_video).pack(side="left")
        self.file_lbl = ttk.Label(top, text="No video selected", foreground="#666")
        self.file_lbl.pack(side="left", padx=12, fill="x", expand=True)
        self.run_btn = ttk.Button(top, text="Analyse", command=self.start, state="disabled")
        self.run_btn.pack(side="right")
        self.cancel_btn = ttk.Button(top, text="Cancel", command=self.cancel_run)
        self.settings_btn = ttk.Button(top, text="Settings ▾", command=self.toggle_settings)
        self.settings_btn.pack(side="right", padx=8)

        self.settings = SettingsPanel(self.root)
        self.settings.load()
        self.settings_shown = False

        prog = ttk.Frame(self.root, padding=(12, 2, 12, 6))
        prog.pack(fill="x")
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=1.0)
        self.progress.pack(fill="x")
        self.status = ttk.Label(prog, text="Open a dashcam video to start.", foreground="#666")
        self.status.pack(anchor="w", pady=(4, 0))
        self.prog_frame = prog

        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        left = ttk.Frame(body)
        self.view = tk.Label(left, bg="#111", fg="#888", text="Video will appear here",
                             width=10, height=5)
        self.view.pack(fill="both", expand=True)
        self.view.bind("<Configure>", lambda e: self._redraw())
        ctl = ttk.Frame(left, padding=(0, 6, 0, 0))
        ctl.pack(fill="x")
        self.play_btn = ttk.Button(ctl, text="▶ Play", width=9, command=self.toggle_play, state="disabled")
        self.play_btn.pack(side="left")
        self.pos = tk.DoubleVar(value=0)
        self.scale = ttk.Scale(ctl, from_=0, to=1, variable=self.pos, command=self._on_scrub)
        self.scale.pack(side="left", fill="x", expand=True, padx=8)
        self.scale.state(["disabled"])
        self.time_lbl = ttk.Label(ctl, text="0:00 / 0:00", width=13, anchor="e")
        self.time_lbl.pack(side="right")
        body.add(left, weight=3)

        right = ttk.Frame(body, padding=(10, 0, 0, 0))
        cols = ("win", "time", "kmh", "status", "method")
        self.table = ttk.Treeview(right, columns=cols, show="headings", height=18, selectmode="browse")
        for c, txt, w, anchor in (("win", "#", 36, "e"), ("time", "Time (s)", 92, "e"),
                                  ("kmh", "km/h", 64, "e"), ("status", "Status", 120, "w"),
                                  ("method", "Measured from", 150, "w")):
            self.table.heading(c, text=txt, anchor=anchor)
            self.table.column(c, width=w, anchor=anchor, stretch=(c == "method"))
        sb = ttk.Scrollbar(right, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=sb.set)
        self.table.bind("<<TreeviewSelect>>", self._on_row)
        btns = ttk.Frame(right, padding=(0, 8, 0, 0))
        btns.pack(side="bottom", fill="x")
        self.summary = ttk.Label(right, text="", foreground="#666", wraplength=380, justify="left")
        self.summary.pack(side="bottom", fill="x", pady=(8, 0))
        sb.pack(side="right", fill="y")
        self.table.pack(fill="both", expand=True)
        self.save_vid_btn = ttk.Button(btns, text="Save video…", command=self.save_video, state="disabled")
        self.save_vid_btn.pack(side="left")
        self.save_csv_btn = ttk.Button(btns, text="Save table (CSV)…", command=self.save_csv, state="disabled")
        self.save_csv_btn.pack(side="left", padx=6)
        self.folder_btn = ttk.Button(btns, text="Open folder", command=self.open_folder, state="disabled")
        self.folder_btn.pack(side="left")
        self.calib_btn = ttk.Button(btns, text="Calibrate…", command=self.calibrate_dialog, state="disabled")
        self.calib_btn.pack(side="right")
        body.add(right, weight=2)

        self.root.bind("<space>", lambda e: self.toggle_play())

    def toggle_settings(self):
        if self.settings_shown:
            self.settings.pack_forget()
            self.settings_btn.configure(text="Settings ▾")
        else:
            self.settings.pack(fill="x", before=self.prog_frame)
            self.settings_btn.configure(text="Settings ▴")
        self.settings_shown = not self.settings_shown

    # ------------------------------------------------------------- choosing
    def choose_video(self):
        if self.worker and self.worker.is_alive():
            return
        p = filedialog.askopenfilename(title="Choose a dashcam video", filetypes=VIDEO_TYPES)
        if p:
            self.set_video(Path(p))

    def set_video(self, p: Path):
        if not p.is_file():
            messagebox.showerror(APP_NAME, f"File not found:\n{p}")
            return
        self.video_path = p
        self.file_lbl.configure(text=p.name, foreground="")
        self.run_btn.configure(state="normal")
        self.status.configure(text="Ready. Press Analyse.")
        # show the first frame as a preview
        cap = cv2.VideoCapture(str(p))
        ok, fr = cap.read()
        cap.release()
        self._close_player()
        if ok:
            self.last_frame = fr
            self._redraw()
        else:
            messagebox.showerror(APP_NAME, "This file could not be read as a video.")
            self.run_btn.configure(state="disabled")

    # ------------------------------------------------------------- running
    def start(self):
        if not self.video_path or (self.worker and self.worker.is_alive()):
            return
        self._close_player()
        self.table.delete(*self.table.get_children())
        self.summary.configure(text="")
        for b in (self.save_vid_btn, self.save_csv_btn, self.folder_btn, self.calib_btn):
            b.configure(state="disabled")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out_dir = results_root() / f"{self.video_path.stem}_{stamp}"
        self.cancel.clear()
        s = self.settings.settings()
        self.settings.save()
        self.used_settings = s
        self.run_btn.pack_forget()
        self.cancel_btn.pack(side="right")
        self.t_start = time.monotonic()
        self.worker = threading.Thread(target=self._work, args=(str(self.video_path), self.out_dir, s),
                                       daemon=True)
        self.worker.start()

    def _work(self, path, out_dir, s):
        def progress(p, msg):
            self.msgs.put(("progress", p, msg))

        def preview(frame):
            with self.preview_lock:
                self.preview_frame = frame

        try:
            res = process_video(path, str(out_dir), s, progress=progress, preview=preview,
                                log=lambda *a: None, cancel=self.cancel)
            self.msgs.put(("done", res))
        except Exception as e:
            if self.cancel.is_set():
                self.msgs.put(("cancelled",))
            else:
                traceback.print_exc()
                self.msgs.put(("error", str(e)))

    def cancel_run(self):
        self.cancel.set()
        self.status.configure(text="Cancelling…")

    def _poll(self):
        try:
            while True:
                m = self.msgs.get_nowait()
                kind = m[0]
                if kind == "progress":
                    _, p, msg = m
                    self.progress.configure(value=p)
                    el = time.monotonic() - self.t_start
                    eta = f" · about {int(el / p - el)} s left" if 0.05 < p < 1 else ""
                    self.status.configure(text=msg + eta)
                elif kind == "done":
                    self._finished(m[1])
                elif kind == "error":
                    self._reset_run_buttons()
                    self.status.configure(text="Error: " + m[1])
                    messagebox.showerror(APP_NAME, m[1])
                elif kind == "cancelled":
                    self._reset_run_buttons()
                    self.progress.configure(value=0)
                    self.status.configure(text="Cancelled.")
                    if self.out_dir and self.out_dir.exists():
                        shutil.rmtree(self.out_dir, ignore_errors=True)
        except queue.Empty:
            pass
        with self.preview_lock:
            fr, self.preview_frame = self.preview_frame, None
        if fr is not None:
            self.last_frame = fr
            self._redraw()
        self.root.after(40, self._poll)

    def _reset_run_buttons(self):
        self.cancel_btn.pack_forget()
        self.run_btn.pack(side="right")

    def _finished(self, res: dict):
        self.result = res
        self._reset_run_buttons()
        took = time.monotonic() - self.t_start
        self.progress.configure(value=1.0)
        self.status.configure(text=f"Done in {took:.0f} s ({res['frames'] / max(took, 1e-3):.0f} frames/s). "
                                   f"Saved to {self.out_dir}")
        for r in res["windows"]:
            kmh = "—" if r["speed_kmh"] is None else f"{r['speed_kmh']:.1f}"
            self.table.insert("", "end", iid=str(r["window"]),
                              values=(r["window"], f"{r['start_s']:.0f}–{r['end_s']:.0f}", kmh,
                                      r["status"], r.get("method", "")))
        c = res["calibration"]
        moving = [r["speed_kmh"] for r in res["windows"] if r["speed_kmh"]]
        avg = f"Average while moving: {np.mean(moving):.1f} km/h. " if moving else ""
        hint = ("" if c.get("scale_factor") else
                " No lane pair to auto-scale from on this road: use Calibrate… once with a known "
                "speed for exact values.")
        if c.get("warning"):
            hint = " ⚠ " + c["warning"]
        if c.get("true_fps"):
            hint += f" (Repeated frames removed: real frame rate {c['true_fps']:g} fps.)"
        self.summary.configure(
            text=f"{avg}Stopped for {res['stationary_s']:.0f} s of {res['duration_s']:.0f} s. "
                 f"Camera height {c['camera_height_m']} m"
                 f"{' (auto-scaled from lane width)' if c.get('scale_factor') else ' (assumed)'}, "
                 f"horizon {c['vanishing_point']}.{hint}")
        for b in (self.save_vid_btn, self.save_csv_btn, self.folder_btn, self.calib_btn):
            b.configure(state="normal")
        self._open_player(self.out_dir / "annotated.mp4")
        if c.get("warning"):
            messagebox.showwarning(APP_NAME, c["warning"])

    # ---------------------------------------------------------------- player
    def _open_player(self, path: Path):
        self._close_player()
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.cap = None
            return
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (self.result or {}).get("frames", 1)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.scale.configure(to=max(self.n_frames - 1, 1))
        self.scale.state(["!disabled"])
        self.play_btn.configure(state="normal")
        self._show(0)

    def _close_player(self):
        self.playing = False
        self.play_btn.configure(text="▶ Play")
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.play_btn.configure(state="disabled")
        self.scale.state(["disabled"])

    def _show(self, idx: int, seek: bool = True):
        if self.cap is None:
            return False
        idx = int(np.clip(idx, 0, max(self.n_frames - 1, 0)))
        if seek:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = self.cap.read()
        if not ok:
            return False
        self.cur = idx
        self.last_frame = fr
        self._redraw()
        self._seeking = True
        self.pos.set(idx)
        self._seeking = False
        self.time_lbl.configure(text=f"{self._fmt(idx / self.fps)} / {self._fmt(self.n_frames / self.fps)}")
        return True

    @staticmethod
    def _fmt(t):
        return f"{int(t // 60)}:{int(t % 60):02d}"

    def _redraw(self):
        if self.last_frame is None:
            return
        w, h = self.view.winfo_width(), self.view.winfo_height()
        if w < 20 or h < 20:
            return
        self.photo = bgr_to_photo(self.last_frame, w, h)
        self.view.configure(image=self.photo, text="")

    def toggle_play(self):
        if self.cap is None:
            return
        self.playing = not self.playing
        self.play_btn.configure(text="❚❚ Pause" if self.playing else "▶ Play")
        if self.playing:
            if self.cur >= self.n_frames - 1:
                self._show(0)
            self._t0 = time.monotonic()
            self._f0 = self.cur
            self._tick()

    def _tick(self):
        if not self.playing or self.cap is None:
            return
        target = self._f0 + int((time.monotonic() - self._t0) * self.fps)
        if target >= self.n_frames:
            self.toggle_play()
            return
        if target > self.cur:
            # read forward (skip frames if the UI fell behind) - no seeking needed
            for _ in range(target - self.cur - 1):
                self.cap.grab()
            self._show(target, seek=False)
        self.root.after(max(int(1000 / self.fps / 2), 5), self._tick)

    def _on_scrub(self, _val):
        if self._seeking or self.cap is None:
            return
        idx = int(float(self.pos.get()))
        if self.playing:
            self._t0, self._f0 = time.monotonic(), idx
        self._show(idx)

    def _on_row(self, _e):
        sel = self.table.selection()
        if not sel or self.cap is None or not self.result:
            return
        row = self.result["windows"][int(sel[0]) - 1]
        idx = int(row["start_s"] * self.fps)
        if self.playing:
            self._t0, self._f0 = time.monotonic(), idx
        self._show(idx)

    # ------------------------------------------------------------ calibration
    def calibrate_dialog(self):
        """Correct the camera field of view from one stretch where the true speed is known.

        Distances along the road (and therefore every speed) scale with the
        focal length, so  f_true = f_used * v_true / v_measured,  i.e.
        tan(FOV_new / 2) = tan(FOV_used / 2) * v_measured / v_true.
        """
        if not self.result:
            return
        moving = [r for r in self.result["windows"] if r["speed_kmh"] and r["status"] in ("moving", "low data")]
        if not moving:
            messagebox.showinfo(APP_NAME, "No window with a measured moving speed to calibrate against.")
            return
        sel = self.table.selection()
        default = next((r for r in moving if sel and str(r["window"]) == sel[0]), moving[0])

        dlg = tk.Toplevel(self.root)
        dlg.title("Calibrate with a known speed")
        dlg.transient(self.root)
        dlg.resizable(False, False)
        frm = ttk.Frame(dlg, padding=16)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Pick a stretch where you know the real speed\n"
                            "(speedometer, GPS, cruise control) and enter it.",
                  justify="left").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        ttk.Label(frm, text="Window").grid(row=1, column=0, sticky="w")
        choices = [f"{r['window']}  ({r['start_s']:.0f}–{r['end_s']:.0f} s, measured {r['speed_kmh']:.1f} km/h)"
                   for r in moving]
        wv = tk.StringVar(value=choices[moving.index(default)])
        ttk.Combobox(frm, textvariable=wv, values=choices, state="readonly", width=40).grid(
            row=1, column=1, sticky="w", pady=4)
        ttk.Label(frm, text="Real speed (km/h)").grid(row=2, column=0, sticky="w")
        sv = tk.StringVar()
        ent = ttk.Entry(frm, textvariable=sv, width=10)
        ent.grid(row=2, column=1, sticky="w", pady=4)
        ent.focus_set()

        def apply():
            try:
                true_kmh = float(sv.get())
                assert true_kmh > 0
            except (ValueError, AssertionError):
                messagebox.showerror(APP_NAME, "Enter the real speed as a number, e.g. 50", parent=dlg)
                return
            row = moving[choices.index(wv.get())]
            used = self.used_settings.hfov_deg
            new = math.degrees(2 * math.atan(math.tan(math.radians(used) / 2) * row["speed_kmh"] / true_kmh))
            new = min(max(new, 30.0), 170.0)
            self.settings.vars["hfov_deg"].set(f"{new:.1f}")
            self.settings.save()
            dlg.destroy()
            if messagebox.askyesno(APP_NAME, f"Camera field of view set to {new:.1f}° (was {used:.1f}°).\n"
                                             "It will be remembered for next time.\n\nRe-analyse this video now?"):
                self.start()

        b = ttk.Frame(frm)
        b.grid(row=3, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(b, text="Cancel", command=dlg.destroy).pack(side="right")
        ttk.Button(b, text="Apply", command=apply).pack(side="right", padx=6)
        dlg.bind("<Return>", lambda e: apply())
        dlg.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - dlg.winfo_reqwidth()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - dlg.winfo_reqheight()) // 3
        dlg.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        dlg.grab_set()

    # ----------------------------------------------------------------- export
    def save_video(self):
        if not self.out_dir:
            return
        dst = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 video", "*.mp4")],
                                           initialfile=f"{self.video_path.stem}_speed.mp4")
        if dst:
            shutil.copyfile(self.out_dir / "annotated.mp4", dst)

    def save_csv(self):
        if not self.out_dir:
            return
        dst = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV table", "*.csv")],
                                           initialfile=f"{self.video_path.stem}_speed_table.csv")
        if dst:
            shutil.copyfile(self.out_dir / "speed_table.csv", dst)

    def open_folder(self):
        if self.out_dir:
            open_in_file_manager(self.out_dir)

    def on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno(APP_NAME, "Analysis is still running. Stop it and quit?"):
                return
            self.cancel.set()
            self.worker.join(timeout=3)
        self._close_player()
        self.root.destroy()


def main():
    if sys.stdout is None or sys.stderr is None:   # windowed app build: keep a log instead
        log = open(results_root() / "roadspeed.log", "a", buffering=1, encoding="utf-8")
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log
    if sys.platform == "win32":                 # crisp text on high-DPI laptop screens
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    icon = Path(__file__).resolve().parent / "assets" / "icon.png"
    if icon.exists():
        try:
            root.iconphoto(True, tk.PhotoImage(file=str(icon)))
        except tk.TclError:
            pass
    root.geometry("1280x760")
    initial = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
    App(root, initial)
    root.mainloop()


if __name__ == "__main__":
    main()
