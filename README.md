# Road Speed Grid

Estimates a car's speed from dashcam video: it finds the road, lays a 1 m grid on it
that bends with the road, tracks the lane markings, times how long they take to cross
consecutive grid lines (speed = 1 m / time), and reports the speed every 3 seconds.

Works on any road: with dashed lane markings it times the dashes; with only solid lines,
or no markings at all, it measures how far the road surface itself (cracks, patches,
grain, the verge) moves across the grid between frames. The table says which was used.

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
2. When it finishes, the table shows one row per 3 seconds: speed, whether the car was
   moving / stopped, and what the speed was **measured from** (lane markings or road
   surface). Click a row to jump the video there. Space = play/pause.
3. **Save video…** / **Save table (CSV)…**, or **Open folder** (results are also kept in
   `Documents\RoadSpeed`).
4. **Fast-forwarded video?** If a clip was sped up (common for road-trip videos online),
   the app warns you (on-screen speeds of 200+ km/h). Set *Settings → Video sped up ×*
   (e.g. 3) and analyse again to get the real speed.
5. **Other vehicles.** In the analysed video every car, truck, bus or motorbike gets a
   magenta rectangle with its **absolute speed** on a tag above it (not in the table).
   A box without a number means the vehicle is too far, or only partly in view, for a
   reliable speed. Untick *Settings → Show other vehicles' speeds* to skip this (faster).
6. **Calibrate… (do this once per dashcam).** Every speed scales with the camera's
   field of view. Pick a stretch where you know the real speed (speedometer, GPS,
   cruise control), enter it, and the app corrects the field of view, remembers it,
   and can re-analyse. Lane-width auto-scaling handles the camera height for you.

Command line (no window): `python cli.py video.mp4 --hfov 110` (add `--no-vehicles` to skip other vehicles)

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
| Stop detection | Near-field road patch registered frame-to-frame with **FFT phase correlation** + frame-difference test, plus "markings tracked but not moving"; overall feature motion across the picture must also be ~0 (stops blurry/compressed video being mistaken for standing still); hysteresis; moving markings or surface veto |
| Road-surface speed (any road) | **Dense:** a strip of road just ahead is re-projected to a fine top-down view every frame; along-road structure (solid lines, worn wheel tracks) is removed and a slowly-updated background takes out anything fixed in the picture (reflections, bonnet, text); the shift between frames is found by **normalised cross-correlation** (coarse-to-fine, sub-pixel), taking the strongest *distinct* peak away from zero so static compression noise can't pull it to 0; slow driving is compared over 4-8 frames. **Sparse:** corner features over the whole ground area tracked with **pyramidal Lucas-Kanade** (forward-backward checked), mapped to the ground plane, robust mode of their travel. The dense estimate is preferred, the sparse one fills gaps |
| Dashed-line pattern | The brightness profile along each dashed lane line (dash, gap, dash … over ~25 m) is matched between frames (NCC, sub-pixel); robust when dashes move a large part of their spacing per frame (fast driving, low frame rate, fast-forwarded video) where following single dashes mixes them up. A line is only used if its profile really looks like dashes, and never beyond a vehicle ahead |
| Repeated frames | Videos converted between frame rates (e.g. 25 → 30 fps) repeat frames; these are detected, skipped, and real frames are timed at the true rate |
| Fusion | Per moment: dash crossings of the grid lines (≥ 4 within ±0.75 s; the pattern match wins if they disagree by > 20 %), else dashed-line pattern motion, else road surface, else a few crossings; the window widens to ±1.5 s when evidence is sparse; stopped = 0 |
| Edge lines | Paint is searched up to 5 m beyond the colour-based road edge; a long solid line there becomes the grid edge (fixes two-tone / worn roads), and the edge stays locked to it through frames where the line isn't seen |
| Crossing validation | A pair of crossings only counts if both happen within 0.6 s, the marking is moving, and the result agrees with its Kalman velocity (±35 %); nothing is counted while stopped |
| Other vehicles | **YOLOX-S** detector (OpenCV DNN, COCO cars/trucks/buses/motorbikes) ~4× per second on the road band, plus a zoomed pass round the vanishing point cut from the full-resolution frame for far vehicles; boxes linked over time by overlap with each track's predicted box. Distance = where the box's bottom edge touches the road, through the same ground-plane homography as the 1 m grid; relative speed = weighted straight-line fit of that distance over ±1 s (±1.5 s for far vehicles), fused with the box's growth rate (d ln w/dt = −v/Z) for whole boxes. **Absolute speed = our speed + relative speed.** A number is only printed when its estimated error is ≤ 6 km/h; grid and marking overlays are cleared inside the boxes |
| Reporting | Per-frame speed timeline (0 when stopped, robust median of nearby samples otherwise, short gaps bridged) → time-averaged per 3 s window with a status: moving / stopped / stopping-starting / low data / no data. The video is drawn in the same pass, a few seconds behind the analysis, so its overlay shows the final numbers (same as the table) |

## Settings

* **Camera field of view** - the one setting that sets the speed scale; use *Calibrate…*.
* **Camera height** - corrected automatically from the lane width (default 3.5 m) when the
  road has two lane lines a lane apart; otherwise 1.3 m is assumed - use *Calibrate…*.
* **Horizon** - auto. Set a % only if the grid looks tilted on a very hilly road.
* **Bonnet** - auto (static rows at the bottom are ignored); set a % to override.
* **Grid lines / Report every** - number of 1 m grid lines, and the table interval.
* **Video sped up ×** - 1 for normal video; e.g. 3 for a 3x fast-forwarded clip.

## Limits

* Needs *something* that moves with the road: dashes, or visible surface texture / verge.
  A perfectly uniform road in a heavily compressed video can leave gaps ("no data").
* Without a lane pair to auto-scale from, absolute speed depends on the camera height
  setting - one *Calibrate…* with a known speed fixes it for that dashcam.
* Assumes a flat road and a normal (not strongly fisheye) lens.
* Braking/pulling-away windows are smoothed over ~1.5 s, so they read a few km/h off.
* Other vehicles' speeds are reliable up to about 50 m; further away (or while a vehicle
  is cut off by the frame edge) only the box is shown. They share the camera scale with
  our own speed, so *Calibrate…* corrects both. Vehicle detection adds roughly 50-80 %
  to the analysis time on a 2-core laptop (less on more cores).

`tools/make_synthetic.py` renders test videos with a known speed: `--stopgo` adds traffic and
a stop; `--scene outback` (only solid edge lines, worn two-tone asphalt, scrub verges) or
`--scene bare` (no markings); `--degrade` (blur + heavy web-video compression), `--bounce`.
