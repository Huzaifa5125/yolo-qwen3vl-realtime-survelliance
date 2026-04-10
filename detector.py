import cv2
import numpy as np
from ultralytics import YOLO
from config import YOLO_MODEL, YOLO_DEVICE


class PersonDetector:
    """YOLOv8s person detector with built-in tracking."""

    PERSON_CLASS_ID = 0  # COCO class 0 = person

    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        self.model.to(YOLO_DEVICE)

    def detect_and_track(self, frame):
        """
        Run detection + tracking on a frame.
        Returns list of dicts:
          [{"track_id": int, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=[self.PERSON_CLASS_ID],
            device=YOLO_DEVICE,
            verbose=False,
            tracker="bytetrack.yaml", # <--- CHANGED FROM botsort.yaml TO bytetrack.yaml
            conf=0.30
        )

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                track_id = (
                    int(boxes.id[i].cpu().numpy()) if boxes.id is not None else -1
                )
                detections.append(
                    {"track_id": track_id, "bbox": bbox, "confidence": conf}
                )

        return detections

    @staticmethod
    def draw_boxes(frame, detections, analysis_map=None):
        """
        Draw bounding boxes + labels on a frame copy.
        analysis_map: optional dict {track_id: analysis_text}
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            tid = det["track_id"]
            conf = det["confidence"]

            # color by track id
            color_hash = (tid * 47) % 255
            color = (
                int(color_hash),
                int((color_hash + 85) % 255),
                int((color_hash + 170) % 255),
            )

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"Person {tid} ({conf:.2f})"
            cv2.putText(
                annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

            # analysis text below box
            if analysis_map and tid in analysis_map:
                text = analysis_map[tid]
                # wrap long text
                y_offset = y2 + 18
                for line in text.split(". "):
                    cv2.putText(
                        annotated, line.strip(),
                        (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
                    )
                    y_offset += 16

        return annotated