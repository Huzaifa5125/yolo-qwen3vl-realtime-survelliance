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

            # always update live frame for streaming
            with self.lock:
                self.live_frame = frame.copy()

            # run YOLO every Nth frame
            if frame_id % DETECTION_EVERY_N == 0 and frame_id != last_processed_id:
                last_processed_id = frame_id
                detections = self.detector.detect_and_track(frame)

                with self.lock:
                    self.live_detections = detections

                if detections:
                    self.buffer.push(frame, detections, frame_id)
            
            time.sleep(0.001)  # yield CPU

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


if __name__ == "__main__":
    pipeline = SurveillancePipeline()
    try:
        pipeline.start()
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down...")
        pipeline.stop()