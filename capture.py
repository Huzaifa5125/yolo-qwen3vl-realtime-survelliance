import cv2
import threading
import time
from config import RTSP_URL, CAPTURE_FPS


class RTSPCapture:
    """Threaded RTSP/webcam frame grabber."""

    def __init__(self, source=None):
        self.source = source or RTSP_URL
        # try int (webcam index) or string (rtsp url)
        try:
            self.source = int(self.source)
        except ValueError:
            pass

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        self.cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return self

    def _read_loop(self):
        target_delay = 1.0 / CAPTURE_FPS
        while self.running:
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                # retry connection
                time.sleep(0.5)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.source)
                continue
            with self.lock:
                self.frame = frame
                self.frame_id += 1
            elapsed = time.time() - t0
            sleep_time = max(0, target_delay - elapsed)
            time.sleep(sleep_time)

    def read(self):
        """Returns (frame_id, frame) or (None, None)."""
        with self.lock:
            if self.frame is None:
                return None, None
            return self.frame_id, self.frame.copy()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)
        self.cap.release()