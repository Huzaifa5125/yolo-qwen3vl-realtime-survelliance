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
        self.total_pushes = 0
        self.total_pops = 0
        self.total_dropped_before_vlm = 0

    def push(self, frame, detections, frame_id):
        """Push a new detected frame (replaces any previous)."""
        ts = time.time()
        with self.lock:
            if self.latest is not None:
                # Previous frame was never consumed by VLM and is replaced now.
                self.total_dropped_before_vlm += 1
            self.total_pushes += 1
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
                if item is not None:
                    self.total_pops += 1
                return item
        return None

    def get_stats(self):
        """Return buffer flow counters used for drop-rate reporting."""
        with self.lock:
            return {
                "buffer_pushes": self.total_pushes,
                "buffer_pops": self.total_pops,
                "buffer_drops_before_vlm": self.total_dropped_before_vlm,
            }
