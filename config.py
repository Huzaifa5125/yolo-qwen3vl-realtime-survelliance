import os

# ─── RTSP / Camera ───
RTSP_URL = os.getenv("RTSP_URL", "http://192.168.228.202/video")  # "0" = webcam fallback
CAPTURE_FPS = 30
DETECTION_EVERY_N = 5  # run YOLO on every Nth frame

# ─── GPU ───
GPU_ID = 2
YOLO_DEVICE = f"cuda:{GPU_ID}"
VLM_DEVICE = f"cuda:{GPU_ID}"

# ─── Models ───
YOLO_MODEL = "yolov8m.pt"
QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# ─── Storage ───
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output1")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
DB_PATH = os.path.join(OUTPUT_DIR, "surveillance.db")

# ─── Web ───
WEB_HOST = "0.0.0.0"
WEB_PORT = 8081

# ─── VLM Generation ───
VLM_MAX_TOKENS = 256
VLM_TOP_P = 0.8
VLM_TOP_K = 20
VLM_TEMPERATURE = 0.7
VLM_REPETITION_PENALTY = 1.0

# Create dirs
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)