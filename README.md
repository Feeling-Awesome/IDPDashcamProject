# Road Speed Grid

Estimates a car's speed from dashcam video: it finds the road, lays a 1 m grid on it
that bends with the road, tracks the lane markings, times how long they take to cross
consecutive grid lines (speed = 1 m / time), and reports the speed every 3 seconds.

## Start the app (Windows)

* **Double-click `Start RoadSpeed.bat`.** The first time it installs what it needs
  (Python must be installed - it tells you where to get it) and puts a
  **Road Speed Grid** icon on your Desktop. From then on just double-click that icon.
* Want a real `.exe` that runs on any PC without Python? Double-click
  **`Build RoadSpeed.exe.bat`** once. It creates `dist\RoadSpeed\RoadSpeed.exe`
  (copy or zip the whole `dist\RoadSpeed` folder to share it).
* Mac: double-click `Start RoadSpeed (Mac).command`.

## Using it

1. **Open video…**, then **Analyse**. You see the video being processed live.
2. When it finishes, the table shows one row per 3 seconds (speed, and whether the car
   was moving / stopped). Click a row to jump the video there. Space = play/pause.
3. **Save video…** / **Save table (CSV)…**, or **Open folder** (results are also kept in
   `Documents\RoadSpeed`).
4. **Calibrate… (do this once per dashcam).** Every speed scales with the camera's
   field of view. Pick a stretch where you know the real speed (speedometer, GPS,
   cruise control), enter it, and the app corrects the field of view, remembers it,
   and can re-analyse. Lane-width auto-scaling handles the camera height for you.

Command line (no window): `python cli.py video.mp4 --hfov 110`

## How it works

| Stage | Technique |
|---|---|
| Camera orientation | Edges of brightness *and* colour saturation with per-frame adaptive Canny thresholds → probabilistic Hough lines on sampled frames → left/right line intersections → weighted 2-D vote for the **vanishing point** → camera pitch & yaw |
| Bonnet / overlay | Rows at the bottom of the frame that never change across the video are excluded automatically |
| Metric ground plane | Pinhole + flat-road model → **ground-plane homography** → bird's-eye view (BEV) at 14 px/m |
| Scale calibration | Measured lateral spacing of lane lines in BEV vs. known lane width (default 3.5 m) → corrects camera height (optional) |
| Road detection | Adaptive Gaussian **YCrCb colour model** of asphalt (at half resolution) (seeded in front of car, updated online), **Mahalanobis** thresholding, morphological closing/opening, connected component containing the car's path |
| Road edges | Each edge followed outward from the car with a **continuity check** (a jump = vehicle/occlusion, stop there) and gated against the last frame → **joint parallel-edge model** (shared heading + curvature, separate offsets) fitted by regularised robust least squares → plausibility checks → a fresh, un-gated detection runs alongside and takes over when it explains the image clearly better (recovery after occlusions) → temporal smoothing |
| Turn-following grid | Centre line parameterised by **arc length**; grid lines placed every 1 m along it, perpendicular to the road, spanning edge to edge; drawn back into the frame with the inverse homography |
| Lane markings | Horizontal **white top-hat** + adaptive MAD threshold, HSV yellow mask, morphology, connected components filtered by shape (minAreaRect elongation) |
| Tracking | Dash endpoints (near/far ends) in road coordinates, **constant-velocity Kalman filter** per endpoint, gated nearest-neighbour assignment (cheapest pairs first) |
| Timing | Crossing time of each grid line **interpolated between frames** (sub-frame accuracy); speed = 1 m / (t<sub>k</sub> − t<sub>k+1</sub>) |
| Free space | Grid is drawn only up to the first vehicle blocking the lane ahead (a vehicle smears far away from the camera in the top-down view, so only long blocked stretches count; small gaps at the frame edge, shadows or cracks are ignored) |
| Stop detection | Near-field road patch registered frame-to-frame with **FFT phase correlation** + frame-difference test, plus "markings tracked but not moving"; hysteresis; moving markings veto |
| Crossing validation | A pair of crossings only counts if both happen within 0.6 s, the marking is moving, and the result agrees with its Kalman velocity (±35 %); nothing is counted while stopped |
| Reporting | Per-frame speed timeline (0 when stopped, robust median of nearby samples otherwise, short gaps bridged) → time-averaged per 3 s window with a status: moving / stopped / stopping-starting / low data / no data. The video is drawn in the same pass, a few seconds behind the analysis, so its overlay shows the final numbers (same as the table) |

## Settings

* **Camera field of view** - the one setting that sets the speed scale; use *Calibrate…*.
* **Camera height** - corrected automatically from the lane width (default 3.5 m).
* **Horizon** - auto. Set a % only if the grid looks tilted on a very hilly road.
* **Bonnet** - auto (static rows at the bottom are ignored); set a % to override.
* **Grid lines / Report every** - number of 1 m grid lines, and the table interval.

## Limits

* Needs **dashed** markings (or any marking with ends) - a solid line has nothing to time.
* Assumes a flat road and a normal (not strongly fisheye) lens.
* Braking/pulling-away windows are smoothed over ~1.5 s, so they read a few km/h off.

`tools/make_synthetic.py` renders test videos with a known speed (`--stopgo` adds traffic and a stop).
