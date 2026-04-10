import os
import cv2
import base64
import json
import threading
import time
import torch
from io import BytesIO
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from config import (
    QWEN_MODEL, VLM_DEVICE, GPU_ID,
    VLM_MAX_TOKENS, VLM_TOP_P, VLM_TOP_K,
    VLM_TEMPERATURE, VLM_REPETITION_PENALTY,
    FRAMES_DIR,
)
from buffer import DetectionBuffer
from detector import PersonDetector
from storage import StorageManager


class VLMAnalyzer:
    """Qwen3-VL based surveillance frame analyzer."""

    # ── Default (general surveillance) prompt ──
    DEFAULT_PROMPT = (
        "You are a surveillance analyst. Analyze this frame with detected persons. "
        "For each person, describe their action in one short line. "
        "Flag anything suspicious (running, fighting, loitering, trespassing, "
        "carrying weapons/unusual objects). "
        "Classify the overall scene as SUSPICIOUS or NORMAL. "
        "Format:\n"
        "P<id>: <action>\n"
        "...\n"
        "Status: <SUSPICIOUS|NORMAL>"
    )

    # ── Built-in preset prompts ──
    PRESET_PROMPTS = {
        "general": DEFAULT_PROMPT,

        "crawling": (
            "You are a surveillance analyst. Focus on detecting anyone CRAWLING, "
            "crouching low to the ground, or moving on hands and knees. "
            "For each detected person, state whether they are crawling or not and "
            "describe their posture in one short line. "
            "Classify the overall scene as SUSPICIOUS if anyone is crawling, otherwise NORMAL. "
            "Format:\n"
            "P<id>: <posture description>\n"
            "...\n"
            "Status: <SUSPICIOUS|NORMAL>"
        ),

        "clothing_color": (
            "You are a surveillance analyst. Focus on identifying the CLOTHING COLOR "
            "of each detected person. Describe the color of their shirt/top, pants/bottom, "
            "and any distinctive accessories. "
            "Flag anyone wearing blue clothing specifically. "
            "Classify the overall scene as SUSPICIOUS if a person matching the clothing "
            "description is found, otherwise NORMAL. "
            "Format:\n"
            "P<id>: <clothing description>\n"
            "...\n"
            "Status: <SUSPICIOUS|NORMAL>"
        ),

        "running": (
            "You are a surveillance analyst. Focus on detecting anyone RUNNING, "
            "sprinting, or moving at high speed. "
            "For each detected person, describe their movement speed and gait. "
            "Classify the overall scene as SUSPICIOUS if anyone is running, otherwise NORMAL. "
            "Format:\n"
            "P<id>: <movement description>\n"
            "...\n"
            "Status: <SUSPICIOUS|NORMAL>"
        ),

        "fighting": (
            "You are a surveillance analyst. Focus on detecting any FIGHTING, "
            "physical altercation, aggressive gestures, pushing, or violent behavior. "
            "For each detected person, describe their body language and interaction "
            "with others. "
            "Classify the overall scene as SUSPICIOUS if any fighting is detected, "
            "otherwise NORMAL. "
            "Format:\n"
            "P<id>: <behavior description>\n"
            "...\n"
            "Status: <SUSPICIOUS|NORMAL>"
        ),

        "loitering": (
            "You are a surveillance analyst. Focus on detecting anyone LOITERING — "
            "standing idle for extended time, lingering near entrances/exits, or "
            "pacing without clear purpose. "
            "For each detected person, describe whether they appear to be loitering "
            "and their apparent intent. "
            "Classify the overall scene as SUSPICIOUS if loitering is detected, "
            "otherwise NORMAL. "
            "Format:\n"
            "P<id>: <activity description>\n"
            "...\n"
            "Status: <SUSPICIOUS|NORMAL>"
        ),

        "weapon": (
            "You are a surveillance analyst. Focus on detecting anyone carrying or "
            "brandishing WEAPONS or weapon-like objects (knives, guns, bats, pipes, "
            "sharp objects). "
            "For each detected person, describe what they are carrying in their hands. "
            "Classify the overall scene as SUSPICIOUS if any weapon is detected, "
            "otherwise NORMAL. "
            "Format:\n"
            "P<id>: <object in hands description>\n"
            "...\n"
            "Status: <SUSPICIOUS|NORMAL>"
        ),

        "trespassing": (
            "You are a surveillance analyst. Focus on detecting anyone who appears to "
            "be TRESPASSING — climbing fences/walls, entering restricted areas, "
            "bypassing barriers, or sneaking. "
            "For each detected person, describe their movement pattern and location "
            "relative to boundaries. "
            "Classify the overall scene as SUSPICIOUS if trespassing is detected, "
            "otherwise NORMAL. "
            "Format:\n"
            "P<id>: <movement/location description>\n"
            "...\n"
            "Status: <SUSPICIOUS|NORMAL>"
        ),

        "abandoned_object": (
            "You are a surveillance analyst. Focus on detecting any ABANDONED OBJECTS "
            "such as unattended bags, packages, or items left behind. "
            "Also note if any person drops something and walks away. "
            "For each detected person, describe what they are carrying or if they "
            "left something behind. "
            "Classify the overall scene as SUSPICIOUS if an abandoned object is found, "
            "otherwise NORMAL. "
            "Format:\n"
            "P<id>: <carrying/drop description>\n"
            "...\n"
            "Status: <SUSPICIOUS|NORMAL>"
        ),
    }

    def __init__(self, buffer: DetectionBuffer, storage: StorageManager):
        self.buffer = buffer
        self.storage = storage
        self.running = False
        self._thread = None

        # ── Runtime prompt state (thread-safe) ──
        self.prompt_lock = threading.Lock()
        self._active_prompt = self.DEFAULT_PROMPT
        self._active_prompt_name = "general"

        # latest analysis result (for web UI)
        self.lock = threading.Lock()
        self.latest_analysis = None
        self.metrics_lock = threading.Lock()
        self.vlm_frames_processed = 0
        self.vlm_total_latency = 0.0
        self.vlm_min_latency = None
        self.vlm_max_latency = None

        print(f"[VLM] Loading Qwen3-VL on cuda:{GPU_ID}...")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL,
            torch_dtype=torch.float16,
            device_map={"": VLM_DEVICE},
        )
        self.processor = AutoProcessor.from_pretrained(QWEN_MODEL)
        print("[VLM] Model loaded.")

    # ── Prompt management API (called from web routes) ──

    def set_prompt(self, prompt_text, prompt_name="custom"):
        """Set the active analysis prompt at runtime."""
        with self.prompt_lock:
            self._active_prompt = prompt_text.strip()
            self._active_prompt_name = prompt_name
        print(f"[VLM] Prompt changed to: '{prompt_name}'")

    def set_preset(self, preset_key):
        """Set the active prompt to a built-in preset."""
        prompt_text = self.PRESET_PROMPTS.get(preset_key)
        if prompt_text is None:
            return False
        self.set_prompt(prompt_text, prompt_name=preset_key)
        return True

    def get_prompt_info(self):
        """Return current prompt name and text (thread-safe)."""
        with self.prompt_lock:
            return {
                "name": self._active_prompt_name,
                "text": self._active_prompt,
                "presets": list(self.PRESET_PROMPTS.keys()),
            }

    def _get_active_prompt(self):
        """Thread-safe read of the current prompt."""
        with self.prompt_lock:
            return self._active_prompt

    # ── Existing methods ──

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _frame_to_base64(self, frame):
        """Convert OpenCV BGR frame to base64 data URI."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def normalize_bbox_to_1000(bbox, frame_height, frame_width):
        """
        Normalize pixel bounding box [x1, y1, x2, y2] to Qwen3-VL's
        0-1000 coordinate space as per official 2D grounding guide.
        """
        x1, y1, x2, y2 = bbox
        nx1 = int(round(x1 / frame_width * 1000))
        ny1 = int(round(y1 / frame_height * 1000))
        nx2 = int(round(x2 / frame_width * 1000))
        ny2 = int(round(y2 / frame_height * 1000))
        nx1 = max(0, min(1000, nx1))
        ny1 = max(0, min(1000, ny1))
        nx2 = max(0, min(1000, nx2))
        ny2 = max(0, min(1000, ny2))
        return [nx1, ny1, nx2, ny2]

    def _build_prompt(self, detections, frame_height, frame_width):
        """
        Build the user prompt with detection context.
        Uses the currently active prompt (preset or custom).
        Bounding boxes are normalized to 0-1000 scale for Qwen3-VL.
        """
        active_prompt = self._get_active_prompt()

        det_info = ""
        for d in detections:
            norm_bbox = self.normalize_bbox_to_1000(
                d["bbox"], frame_height, frame_width
            )
            det_info += (
                f"- Person {d['track_id']}: bbox=[{norm_bbox[0]}, {norm_bbox[1]}, "
                f"{norm_bbox[2]}, {norm_bbox[3]}], "
                f"conf={d['confidence']:.2f}\n"
            )
        return (
            f"{active_prompt}\n\n"
            f"Detected persons:\n{det_info}"
        )

    def analyze(self, frame, detections):
        """Run VLM inference on a single frame."""
        image_uri = self._frame_to_base64(frame)
        h, w = frame.shape[:2]
        user_prompt = self._build_prompt(detections, h, w)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=VLM_MAX_TOKENS,
                top_p=VLM_TOP_P,
                top_k=VLM_TOP_K,
                temperature=VLM_TEMPERATURE,
                repetition_penalty=VLM_REPETITION_PENALTY,
            )

        generated_ids_trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0] if output_text else ""

    def _loop(self):
        """Main analysis loop — always picks up the latest buffered frame."""
        while self.running:
            item = self.buffer.pop(timeout=1.0)
            if item is None:
                continue

            frame = item["frame"]
            detections = item["detections"]
            frame_id = item["frame_id"]
            timestamp = item["timestamp"]

            if not detections:
                continue

            # Read current prompt name for logging & DB
            prompt_info = self.get_prompt_info()
            prompt_name = prompt_info["name"]

            print(
                f"[VLM] Analyzing frame {frame_id} with {len(detections)} persons "
                f"(prompt: {prompt_name})..."
            )
            t0 = time.time()
            analysis_text = self.analyze(frame, detections)
            elapsed = time.time() - t0
            print(f"[VLM] Done in {elapsed:.1f}s: {analysis_text[:120]}...")
            with self.metrics_lock:
                self.vlm_frames_processed += 1
                self.vlm_total_latency += elapsed
                if self.vlm_min_latency is None or elapsed < self.vlm_min_latency:
                    self.vlm_min_latency = elapsed
                if self.vlm_max_latency is None or elapsed > self.vlm_max_latency:
                    self.vlm_max_latency = elapsed

            # classify
            classification = (
                "suspicious"
                if "STATUS: SUSPICIOUS" in analysis_text.upper()
                else "normal"
            )

            # ── Save CLEAN (unannotated) frame ──
            clean_fname = f"frame_{frame_id}_{int(timestamp)}_clean.jpg"
            clean_fpath = os.path.join(FRAMES_DIR, clean_fname)
            cv2.imwrite(clean_fpath, frame)

            # ── Save ANNOTATED frame ──
            annotated = PersonDetector.draw_boxes(frame, detections)
            y = 30
            for line in analysis_text.split("\n"):
                cv2.putText(
                    annotated, line.strip(), (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                )
                y += 20

            ann_fname = f"frame_{frame_id}_{int(timestamp)}.jpg"
            ann_fpath = os.path.join(FRAMES_DIR, ann_fname)
            cv2.imwrite(ann_fpath, annotated)

            # ── Build detections JSON with original pixel bboxes ──
            detections_for_db = [
                {
                    "track_id": d["track_id"],
                    "bbox": d["bbox"],
                    "confidence": round(d["confidence"], 4),
                }
                for d in detections
            ]

            # persist to DB
            track_ids = [d["track_id"] for d in detections]
            self.storage.save_analysis(
                frame_id=frame_id,
                timestamp=timestamp,
                frame_path=ann_fpath,
                clean_frame_path=clean_fpath,
                analysis_text=analysis_text,
                classification=classification,
                person_ids=track_ids,
                num_persons=len(detections),
                detections_json=detections_for_db,
                vlm_latency=elapsed,
            )

            # update latest for web UI
            with self.lock:
                self.latest_analysis = {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "text": analysis_text,
                    "classification": classification,
                    "person_ids": track_ids,
                    "frame_path": ann_fpath,
                    "clean_frame_path": clean_fpath,
                    "detections": detections_for_db,
                    "prompt_used": prompt_name,
                    "vlm_latency": elapsed,
                }

    def get_latest(self):
        with self.lock:
            return self.latest_analysis

    def get_metrics(self):
        with self.metrics_lock:
            if self.vlm_frames_processed > 0:
                avg_latency = self.vlm_total_latency / self.vlm_frames_processed
            else:
                avg_latency = 0.0
            return {
                "vlm_frames_processed": self.vlm_frames_processed,
                "vlm_total_latency": self.vlm_total_latency,
                "vlm_avg_latency": avg_latency,
                "vlm_min_latency": self.vlm_min_latency,
                "vlm_max_latency": self.vlm_max_latency,
            }

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=10)
