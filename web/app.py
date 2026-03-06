import cv2
import json
import time
from io import BytesIO
from flask import (
    Flask, render_template, Response, jsonify, request, send_file,
)
from config import WEB_HOST, WEB_PORT


def create_app(pipeline):
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # ─── Pages ───
    @app.route("/")
    def index():
        return render_template("index.html")

    # ─── MJPEG Live Stream ───
    def gen_frames():
        while True:
            frame = pipeline.get_live_frame()
            if frame is None:
                time.sleep(0.03)
                continue
            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
            time.sleep(1.0 / 30)

    @app.route("/video_feed")
    def video_feed():
        return Response(
            gen_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    # ─── API: Latest Analysis ───
    @app.route("/api/latest_analysis")
    def latest_analysis():
        result = pipeline.analyzer.get_latest()
        if result is None:
            return jsonify({"status": "waiting", "text": "No analysis yet..."})
        return jsonify({
            "status": "ok",
            "frame_id": result["frame_id"],
            "timestamp": result["timestamp"],
            "text": result["text"],
            "classification": result["classification"],
            "person_ids": result["person_ids"],
            "detections": result.get("detections", []),
            "prompt_used": result.get("prompt_used", "general"),
        })

    # ─── API: Get / Set Active Prompt ───
    @app.route("/api/prompt", methods=["GET"])
    def get_prompt():
        """Return the currently active prompt info + list of presets."""
        info = pipeline.analyzer.get_prompt_info()
        return jsonify(info)

    @app.route("/api/prompt", methods=["POST"])
    def set_prompt():
        """
        Set the active analysis prompt.
        Body JSON options:
          { "preset": "running" }          — use a built-in preset
          { "custom": "Your prompt..." }   — use a fully custom prompt
        """
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400

        # Option 1: preset key
        preset_key = data.get("preset")
        if preset_key:
            ok = pipeline.analyzer.set_preset(preset_key)
            if not ok:
                avail = list(pipeline.analyzer.PRESET_PROMPTS.keys())
                return jsonify({
                    "error": f"Unknown preset '{preset_key}'",
                    "available_presets": avail,
                }), 400
            info = pipeline.analyzer.get_prompt_info()
            return jsonify({"status": "ok", "prompt": info})

        # Option 2: custom prompt text
        custom_text = data.get("custom")
        if custom_text and custom_text.strip():
            pipeline.analyzer.set_prompt(custom_text.strip(), prompt_name="custom")
            info = pipeline.analyzer.get_prompt_info()
            return jsonify({"status": "ok", "prompt": info})

        return jsonify({"error": "Provide 'preset' or 'custom' in JSON body"}), 400

    # ─── API: Query Saved Analyses ───
    @app.route("/api/analyses")
    def get_analyses():
        classification = request.args.get("classification")
        start_ts = request.args.get("start_ts", type=float)
        end_ts = request.args.get("end_ts", type=float)
        limit = request.args.get("limit", 50, type=int)
        offset = request.args.get("offset", 0, type=int)

        rows = pipeline.storage.query_analyses(
            classification=classification,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=limit,
            offset=offset,
        )
        return jsonify({"analyses": rows})

    # ─── API: Serve saved frame image ───
    @app.route("/api/frame/<path:frame_path>")
    def serve_frame(frame_path):
        if not frame_path.startswith("/"):
            frame_path = "/" + frame_path
        return send_file(frame_path, mimetype="image/jpeg")

    # ─── API: Serve clean frame with optional bbox overlay ───
    @app.route("/api/frame_with_boxes/<int:analysis_id>")
    def frame_with_boxes(analysis_id):
        show_boxes = request.args.get("show_boxes", "0") == "1"

        rows = pipeline.storage.query_analyses(limit=9999)
        row = None
        for r in rows:
            if r["id"] == analysis_id:
                row = r
                break

        if row is None:
            return jsonify({"error": "Analysis not found"}), 404

        clean_path = row.get("clean_frame_path")
        if not clean_path:
            frame_path = row.get("frame_path", "")
            if not frame_path.startswith("/"):
                frame_path = "/" + frame_path
            return send_file(frame_path, mimetype="image/jpeg")

        if not clean_path.startswith("/"):
            clean_path = "/" + clean_path

        if not show_boxes:
            return send_file(clean_path, mimetype="image/jpeg")

        frame = cv2.imread(clean_path)
        if frame is None:
            return jsonify({"error": "Frame file not found"}), 404

        detections = row.get("detections_json", [])
        if isinstance(detections, str):
            try:
                detections = json.loads(detections)
            except (json.JSONDecodeError, TypeError):
                detections = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            tid = det.get("track_id", -1)
            conf = det.get("confidence", 0.0)
            color_hash = (tid * 47) % 255
            color = (
                int(color_hash),
                int((color_hash + 85) % 255),
                int((color_hash + 170) % 255),
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Person {tid} ({conf:.2f})"
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ret:
            return jsonify({"error": "Encoding failed"}), 500

        bio = BytesIO(buf.tobytes())
        bio.seek(0)
        return send_file(bio, mimetype="image/jpeg")

    # ─── API: Stats ───
    @app.route("/api/stats")
    def stats():
        return jsonify(pipeline.storage.get_stats())

    return app