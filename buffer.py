import threading
import time


class DetectionBuffer:
    """
    Thread-safe buffer holding detected frames for VLM analysis.
    Only keeps the latest entry so the VLM always processes the most recent frame.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None  # (timestamp, frame, detections, frame_id)
        self.event = threading.Event()

    def push(self, frame, detections, frame_id):
        """Push a new detected frame (replaces any previous)."""
        ts = time.time()
        with self.lock:
            self.latest = {
                "timestamp": ts,
                "frame": frame.copy(),
                "detections": detections,
                "frame_id": frame_id,
            }
        self.event.set()

    def pop(self, timeout=1.0):
        """
        Wait for and pop the latest frame.
        Returns dict or None on timeout.
        """
        if self.event.wait(timeout=timeout):
            with self.lock:
                item = self.latest
                self.latest = None
                self.event.clear()
                return item
        return None