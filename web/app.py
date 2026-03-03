import cv2
import time
from flask import Flask, render_template, Response, jsonify, request, send_file
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
            time.sleep(1.0 / 30)  # ~30fps

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
        })

    # ─── API: Query Saved Analyses ───
    @app.route("/api/analyses")
    def get_analyses():
        classification = request.args.get("classification")  # suspicious/normal
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

    # ─── API: Stats ───
    @app.route("/api/stats")
    def stats():
        return jsonify(pipeline.storage.get_stats())

    return app