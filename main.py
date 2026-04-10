import time
import threading
import cv2
from config import DETECTION_EVERY_N, WEB_PORT
from capture import RTSPCapture
from detector import PersonDetector
from buffer import DetectionBuffer
from analyzer import VLMAnalyzer
from storage import StorageManager


class SurveillancePipeline:
    """Orchestrates the full surveillance pipeline."""

    def __init__(self):
        self.capture = RTSPCapture()
        self.detector = PersonDetector()
        self.buffer = DetectionBuffer()
        self.storage = StorageManager()
        self.analyzer = VLMAnalyzer(self.buffer, self.storage)

        # shared state for live feed (web UI reads this)
        self.lock = threading.Lock()
        self.live_frame = None
        self.live_detections = []
        self.running = False
        self.total_frames_seen = 0
        self.yolo_runs = 0
        self.yolo_total_latency = 0.0
        self.yolo_min_latency = None
        self.yolo_max_latency = None
        self.yolo_frames_with_detections = 0

    def start(self):
        self.running = True
        self.capture.start()
        self.analyzer.start()

        # start detection loop
        self._det_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._det_thread.start()

        # start web server (import here to avoid circular)
        from web.app import create_app
        app = create_app(self)
        print(f"[WEB] Starting web UI on http://0.0.0.0:{WEB_PORT}")
        app.run(host="0.0.0.0", port=WEB_PORT, threaded=True, use_reloader=False)

    def _detection_loop(self):
        """Main loop: grab frames, run YOLO every Nth frame, push to buffer."""
        last_processed_id = -1
        while self.running:
            frame_id, frame = self.capture.read()
            if frame is None:
                time.sleep(0.01)
                continue

            self.total_frames_seen += 1

            # always update live frame for streaming
            with self.lock:
                self.live_frame = frame.copy()

            # run YOLO every Nth frame
            if frame_id % DETECTION_EVERY_N == 0 and frame_id != last_processed_id:
                last_processed_id = frame_id
                yolo_t0 = time.time()
                detections = self.detector.detect_and_track(frame)
                yolo_elapsed = time.time() - yolo_t0
                self.yolo_runs += 1
                self.yolo_total_latency += yolo_elapsed
                if self.yolo_min_latency is None or yolo_elapsed < self.yolo_min_latency:
                    self.yolo_min_latency = yolo_elapsed
                if self.yolo_max_latency is None or yolo_elapsed > self.yolo_max_latency:
                    self.yolo_max_latency = yolo_elapsed

                with self.lock:
                    self.live_detections = detections

                if detections:
                    self.yolo_frames_with_detections += 1
                    self.buffer.push(frame, detections, frame_id)
                self.storage.upsert_pipeline_metrics(self.get_pipeline_metrics())
            
            time.sleep(0.001)  # yield CPU

    def get_pipeline_metrics(self):
        """Aggregate runtime metrics for detector, buffer, and analyzer."""
        if self.yolo_runs > 0:
            yolo_avg = self.yolo_total_latency / self.yolo_runs
        else:
            yolo_avg = 0.0
        metrics = {
            "total_frames_seen": self.total_frames_seen,
            "yolo_runs": self.yolo_runs,
            "yolo_total_latency": self.yolo_total_latency,
            "yolo_avg_latency": yolo_avg,
            "yolo_min_latency": self.yolo_min_latency,
            "yolo_max_latency": self.yolo_max_latency,
            "yolo_frames_with_detections": self.yolo_frames_with_detections,
        }
        metrics.update(self.buffer.get_stats())
        metrics.update(self.analyzer.get_metrics())
        return metrics

    def get_live_frame(self):
        """Get latest frame with YOLO boxes drawn (for MJPEG stream)."""
        with self.lock:
            if self.live_frame is None:
                return None
            return PersonDetector.draw_boxes(self.live_frame, self.live_detections)

    def stop(self):
        self.running = False
        self.capture.stop()
        self.analyzer.stop()
        self.storage.upsert_pipeline_metrics(self.get_pipeline_metrics())


if __name__ == "__main__":
    pipeline = SurveillancePipeline()
    try:
        pipeline.start()
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down...")
        pipeline.stop()
