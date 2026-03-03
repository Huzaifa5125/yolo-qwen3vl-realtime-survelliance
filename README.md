# YOLO-Qwen3-VL Surveillance Pipeline

An intelligent surveillance pipeline that combines the real-time object detection capabilities of YOLOv8 with the advanced multi-modal visual language understanding of Qwen3-VL. 

This project captures an RTSP stream (or local webcam), detects people or objects using YOLO, buffers frames containing detections, and processes them asynchronously using a Vision-Language Model (Qwen3-VL) to extract detailed descriptive insights or perform security analysis. It also features a live Flask-based Web UI for streaming the analyzed feed.

## Features

- **Real-Time Video Capture**: Reads from RTSP streams or local webcams.
- **YOLOv8 Object Detection**: High-speed, accurate person/object tracking.
- **Qwen3-VL Integration**: Advanced multi-modal analysis on detected frames to extract intent, description, and scene understanding.
- **Asynchronous Pipeline**: The pipeline decouples video decoding, YOLO detection, and VLM analysis using threaded buffers for optimal performance.
- **Web UI**: Real-time MJPEG streaming dashboard with bounding boxes and VLM insights.

## Project Structure

- `main.py` - Orchestrator that initializes the pipeline components and starts the loop/web server.
- `config.py` - Configuration parameters (Models, RTSP URL, GPU ID, etc).
- `capture.py` - Video capture handler for RTSP/webcams.
- `detector.py` - YOLOv8 object detection wrapper.
- `buffer.py` - Thread-safe queue to pass frames from the detector to the VLM.
- `analyzer.py` - VLM (Qwen3-VL) handler for image understanding and prompting.
- `storage.py` - Handles saving frames and SQLite database logging.
- `web/` - Flask Web UI (dashboard, API endpoints, MJPEG streaming).

## Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for YOLO and Qwen3-VL).

### Install Dependencies

Install the requirements using `pip`:

```bash
pip install -r requirements.txt
```

*(Optional)* You may also want to ensure you have the appropriate `torch` version installed for your CUDA toolkit. 

## Configuration

Edit `config.py` to fit your setup:
- `RTSP_URL`: The RTSP stream link, or `"0"` for your local webcam.
- `GPU_ID`: GPU index to be used.
- `YOLO_MODEL`: The YOLO model file (`yolov8s.pt` by default).
- `QWEN_MODEL`: The Hugging Face repo for the Qwen VL model.

## Usage

Start the surveillance pipeline:

```bash
python main.py
```

Once running, you can access the live Web UI dashboard at `http://localhost:8081` (or the configured `WEB_PORT`).

## License
MIT
