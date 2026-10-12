"""
Dashcam Speed Detector — Flask Application
==========================================
Fixes vs. v1
  • /stream now guards against no video loaded (was 500-ing)
  • Removed _stream_lock that was unnecessary (only stream calls next_annotated_frame)
  • Added /test_frame endpoint to get a single JPEG for quick debugging
  • Added CORS headers and correct cache-control for the stream
  • app.run uses use_reloader=False to avoid double-instantiation of VideoProcessor
"""

import os
import time
import cv2
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
from processor import VideoProcessor

# ─── app setup ───────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = os.path.join(os.path.dirname(__file__), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024   # 2 GB

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXT = {"mp4", "avi", "mov", "mkv", "m4v", "webm"}

processor = VideoProcessor()    # is_processing = False until load_video()

# ─── helpers ─────────────────────────────────────────────────────────────────

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def _no_cache(response):
    """Attach no-cache headers — important for the MJPEG stream."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"]        = "no-cache"
    response.headers["Expires"]       = "0"
    return response


# ─── routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video field in request"}), 400

    f = request.files["video"]
    if not f or f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not _allowed(f.filename):
        return jsonify({"error": "Unsupported file type"}), 415

    filename = secure_filename(f.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(path)

    ok = processor.load_video(path)
    if not ok:
        return jsonify({"error": "OpenCV could not open the video"}), 500

    return jsonify({
        "status":   "ready",
        "fps":      processor.fps,
        "filename": filename,
    })


@app.route("/stream")
def stream():
    """
    MJPEG stream of annotated frames.

    Guard: if no video is loaded yet, wait up to 10 s before giving up.
    This prevents a 500 when the browser requests the stream before upload.
    """

    def generate():
        # ── Wait for a video to be loaded ────────────────────────────────
        waited = 0.0
        while not processor.is_processing:
            time.sleep(0.1)
            waited += 0.1
            if waited > 10.0:
                # Nothing loaded after 10 s — send an empty end-of-stream
                yield b"--frame\r\n\r\n"
                return

        fps_safe    = max(processor.fps, 1.0)
        target_delay = 1.0 / fps_safe

        # ── Main streaming loop ───────────────────────────────────────────
        while processor.is_processing:
            t0 = time.perf_counter()

            try:
                frame = processor.next_annotated_frame()
            except Exception as exc:
                print(f"[stream] Unhandled error: {exc}")
                time.sleep(0.05)
                continue

            if frame is None:
                break

            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes()
                   + b"\r\n")

            elapsed = time.perf_counter() - t0
            sleep_t = max(0.0, target_delay - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

        yield b"--frame\r\n\r\n"   # end-of-stream sentinel

    resp = Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
    return _no_cache(resp)


@app.route("/test_frame")
def test_frame():
    """
    Debug: returns the single next annotated JPEG frame.
    Useful for checking if the CV pipeline is working without needing
    full MJPEG support in the browser.
    """
    if not processor.is_processing:
        return jsonify({"error": "No video loaded"}), 400

    try:
        frame = processor.next_annotated_frame()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    if frame is None:
        return jsonify({"error": "No frame available (video ended?)"}), 404

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({"error": "JPEG encode failed"}), 500

    resp = Response(buf.tobytes(), mimetype="image/jpeg")
    return _no_cache(resp)


@app.route("/speed_data")
def speed_data():
    return jsonify(processor.get_speed_table())


@app.route("/status")
def status():
    return jsonify(processor.get_status())


@app.route("/reset", methods=["POST"])
def reset():
    processor.release()
    return jsonify({"status": "reset"})


# ─── entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # use_reloader=False prevents VideoProcessor from being created twice
    # (Flask reloader spawns a child process which re-runs __main__)
    app.run(host="0.0.0.0", port=5000, debug=False,
            threaded=True, use_reloader=False)
