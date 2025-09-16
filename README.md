# YOLO Detection API

A FastAPI-based microservice for running YOLOv8 object detection. Includes optional training script and deployment configuration.

## Features
- Fast inference with YOLOv8 (ultralytics)
- File upload endpoint (`/api/detect`)
- Health check endpoint (`/api/health`)
- Simple model hot-swap via `app/model/best.pt`
- Deployable to Render / Heroku (Procfile included)

## Project Structure
```
app/
  main.py           # FastAPI app factory and entry
  routes.py         # API endpoints
  services/detection.py
  utils/preprocess.py
  model/            # Place model weights here (best.pt or yolov8n.pt)
requirements.txt
Procfile
runtime.txt
smoke_test.py
render.yaml
```

## Setup
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Locally
```bash
uvicorn app.main:app --reload --port 8000
```
Visit: http://localhost:8000/docs for interactive Swagger UI.

## Endpoints
### GET /api/health
Returns `{ "status": "ok" }` without loading the model (safe liveness probe).

### POST /api/detect
Multipart form with a `file` field (image). Returns JSON like:
```json
{
  "garbage_detected": true,
  "detections": 3,
  "max_confidence": 0.91,
  "confidence_threshold": 0.25,
  "resized": true
}
```

## Model Weights
Place a custom model at `app/model/best.pt` (preferred) or `app/model/yolov8n.pt`. If missing, the default `yolov8n.pt` will auto-download at first inference.


## Deployment (Heroku/Render)
The `Procfile` runs:
```
web: uvicorn app.main:app --host=0.0.0.0 --port=${PORT:-8000}
```

### Render
Use the included `render.yaml` to create a Web Service. It installs requirements and optionally pre-caches the model during build via `scripts/precache_model.py`.

