# Dashcam Highway Speed Detector (DashSpeed)

A computer vision web application to detect the road, project a flexible perspective-aware 1-meter grid overlay, detect and track lane markings via Lucas-Kanade optical flow, and calculate vehicle speed based on the crossing time of consecutive horizontal gridlines.

## Features

- **Road & Lane Boundary Detection**: Uses Hough line transforms and Canny edge detection to automatically locate lane boundaries and vanishing points.
- **Inverse Perspective Mapping (IPM / Bird's-Eye View)**: Transforms camera frames into a metric top-down perspective.
- **Flexible 1-Meter Grid Pattern**:
  - Two vertical guidelines aligning with road boundaries.
  - Horizontal gridlines projected at precise 1-meter intervals on the road surface.
- **Lane Marking Optical Flow Tracking**: Sparse Lucas-Kanade optical flow tracking with HLS color masking (for white/yellow road markings).
- **Speed Calculation**:
  - Calculates vehicle speed from the time ($\Delta t$) taken by lane markings to cross consecutive 1-meter gridlines:
    $$\text{Speed (m/s)} = \frac{1\text{ m}}{\Delta t}, \quad \text{Speed (km/h)} = \frac{3.6}{\Delta t}$$
  - Aggregates readings every 3 seconds and presents them in a real-time tabular format.
- **Minimalistic Dark UI**: Live MJPEG video stream with telemetry HUD and clean monospace data table.

## Project Structure

```
├── app.py              # Flask server and streaming endpoints
├── processor.py        # Core CV algorithms (IPM, lane tracking, grid calculation)
├── requirements.txt    # Python dependencies
├── static/
│   ├── css/style.css   # Minimalistic dark theme
│   └── js/app.js       # Client upload & live polling logic
├── templates/
│   └── index.html      # Minimal web frontend
└── uploads/            # Temporary storage for uploaded dashcam videos
```

## Getting Started

### 1. Installation

Ensure you have Python installed. Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 3. Usage

1. Click **Choose video** and select your dashcam video file (`.mp4`, `.avi`, `.mov`, etc.).
2. Click **Analyse** to start processing.
3. Observe the live stream with gridlines, tracking markers, real-time HUD, and speed data updating every 3 seconds in the table.
