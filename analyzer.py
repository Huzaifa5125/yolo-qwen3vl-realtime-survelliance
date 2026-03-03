import os
import cv2
import base64
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

    SURVEILLANCE_PROMPT = (
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

    def __init__(self, buffer: DetectionBuffer, storage: StorageManager):
        self.buffer = buffer
        self.storage = storage
        self.running = False
        self._thread = None

        # latest analysis result (for web UI)
        self.lock = threading.Lock()
        self.latest_analysis = None

        print(f"[VLM] Loading Qwen3-VL on cuda:{GPU_ID}...")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL,
            torch_dtype=torch.float16,
            device_map={"": VLM_DEVICE},
        )
        self.processor = AutoProcessor.from_pretrained(QWEN_MODEL)
        print("[VLM] Model loaded.")

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

    def _build_prompt(self, detections):
        """Build the user prompt with detection context."""
        det_info = ""
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            det_info += (
                f"- Person {d['track_id']}: bbox=({x1},{y1},{x2},{y2}), "
                f"conf={d['confidence']:.2f}\n"
            )
        return (
            f"{self.SURVEILLANCE_PROMPT}\n\n"
            f"Detected persons:\n{det_info}"
        )

    def analyze(self, frame, detections):
        """Run VLM inference on a single frame."""
        image_uri = self._frame_to_base64(frame)
        user_prompt = self._build_prompt(detections)

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

            print(f"[VLM] Analyzing frame {frame_id} with {len(detections)} persons...")
            t0 = time.time()
            analysis_text = self.analyze(frame, detections)
            elapsed = time.time() - t0
            print(f"[VLM] Done in {elapsed:.1f}s: {analysis_text[:120]}...")

            # classify
            classification = (
                "suspicious"
                if "STATUS: SUSPICIOUS" in analysis_text.upper()
                else "normal"
            )

            # draw annotated frame
            annotated = PersonDetector.draw_boxes(frame, detections)
            # add analysis text overlay
            y = 30
            for line in analysis_text.split("\n"):
                cv2.putText(
                    annotated, line.strip(), (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                )
                y += 20

            # save frame
            fname = f"frame_{frame_id}_{int(timestamp)}.jpg"
            fpath = os.path.join(FRAMES_DIR, fname)
            cv2.imwrite(fpath, annotated)

            # persist to DB
            track_ids = [d["track_id"] for d in detections]
            self.storage.save_analysis(
                frame_id=frame_id,
                timestamp=timestamp,
                frame_path=fpath,
                analysis_text=analysis_text,
                classification=classification,
                person_ids=track_ids,
                num_persons=len(detections),
            )

            # update latest for web UI
            with self.lock:
                self.latest_analysis = {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "text": analysis_text,
                    "classification": classification,
                    "person_ids": track_ids,
                    "frame_path": fpath,
                }

    def get_latest(self):
        with self.lock:
            return self.latest_analysis

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=10)