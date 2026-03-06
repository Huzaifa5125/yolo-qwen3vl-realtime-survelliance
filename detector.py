import cv2
import numpy as np
from ultralytics import YOLO
from config import YOLO_MODEL, YOLO_DEVICE


class PersonDetector:
    """YOLOv8s person detector with built-in tracking."""

    PERSON_CLASS_ID = 0  # COCO class 0 = person

    # ─── Duplicate suppression thresholds ───
    # IoU threshold: if two boxes overlap this much, consider merging
    IOU_MERGE_THRESH = 0.3
    # Containment threshold: if smaller box is this % inside larger box
    CONTAINMENT_THRESH = 0.6
    # Area ratio threshold: only merge if smaller_area / larger_area >= this
    # Prevents merging a far-away person contained inside a close-up person
    AREA_RATIO_THRESH = 0.2

    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        self.model.to(YOLO_DEVICE)

    @staticmethod
    def _box_area(box):
        """Compute area of an [x1, y1, x2, y2] box."""
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    @staticmethod
    def _compute_overlap(box_a, box_b):
        """
        Compute IoU, containment, and area ratio between two boxes.
        Returns (iou, containment_of_smaller, area_ratio).
        """
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0, 0.0, 0.0

        area_a = PersonDetector._box_area(box_a)
        area_b = PersonDetector._box_area(box_b)
        union = area_a + area_b - inter

        iou = inter / union if union > 0 else 0.0

        # containment: what fraction of the SMALLER box is inside the overlap
        smaller_area = min(area_a, area_b)
        larger_area = max(area_a, area_b)
        containment = inter / smaller_area if smaller_area > 0 else 0.0

        # area ratio: how similar in size are the two boxes
        # 1.0 = identical size, 0.01 = one is 100x bigger
        area_ratio = smaller_area / larger_area if larger_area > 0 else 0.0

        return iou, containment, area_ratio

    @staticmethod
    def _suppress_duplicates(detections, iou_thresh, containment_thresh,
                             area_ratio_thresh):
        """
        Remove duplicate detections of the same person.

        Merges two boxes ONLY when ALL conditions are met:
          1. High IoU OR high containment (boxes significantly overlap)
          2. Similar box sizes (area_ratio >= threshold)

        Condition #2 prevents merging a far-away person whose box happens
        to fall inside a close-up person's large bounding box.

        When merged, we keep the LARGER box and the higher-confidence
        track ID.
        """
        if len(detections) <= 1:
            return detections

        # sort by box area descending (largest first)
        detections = sorted(
            detections,
            key=lambda d: PersonDetector._box_area(d["bbox"]),
            reverse=True,
        )

        keep = []
        suppressed = set()

        for i in range(len(detections)):
            if i in suppressed:
                continue

            det_i = detections[i]

            for j in range(i + 1, len(detections)):
                if j in suppressed:
                    continue

                det_j = detections[j]
                iou, containment, area_ratio = PersonDetector._compute_overlap(
                    det_i["bbox"], det_j["bbox"]
                )

                # Must overlap significantly AND be similar in size
                overlap_ok = (iou >= iou_thresh or
                              containment >= containment_thresh)
                size_ok = area_ratio >= area_ratio_thresh

                if overlap_ok and size_ok:
                    # det_j is the smaller box (sorted desc).
                    # Keep det_i (larger). Inherit higher-conf track_id.
                    if det_j["confidence"] > det_i["confidence"]:
                        det_i["track_id"] = det_j["track_id"]
                    suppressed.add(j)

            keep.append(det_i)

        return keep

    def detect_and_track(self, frame):
        """
        Run detection + tracking on a frame.
        Returns list of dicts:
          [{"track_id": int, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]

        Post-processing: suppress duplicate boxes where YOLO detects
        the same person twice (e.g. full body + upper body when seated),
        while preserving separate people at different depths.
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=[self.PERSON_CLASS_ID],
            device=YOLO_DEVICE,
            verbose=False,
            tracker="botsort.yaml",
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

        # ── Suppress duplicate/overlapping boxes ──
        detections = self._suppress_duplicates(
            detections,
            self.IOU_MERGE_THRESH,
            self.CONTAINMENT_THRESH,
            self.AREA_RATIO_THRESH,
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