# YOLO Detection API

A FastAPI-based microservice for running YOLOv8 object detection. Includes optional training script and deployment configuration.

## Features
- Fast inference with YOLOv8 (ultralytics)
- File upload endpoint (`/api/detect`)
- Health check endpoint (`/api/health`)
- Simple model hot-swap via `app/models/yolov8n.pt`
- Training script to fine-tune a model (`training/train.py`)
- Deployable to Render / Heroku (Procfile included)

## Project Structure
```
app/
  main.py           # FastAPI app factory and entry
  routes.py         # API endpoints
  services/detection.py
  utils/preprocess.py
  models/           # Place model weights here (yolov8n.pt)
training/train.py   # Fine-tune script
requirements.txt
Procfile
runtime.txt
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
Returns basic status.

### POST /api/detect
Multipart form with an `file` field (image). Returns JSON with detected objects.
Example using `curl`:
```bash
curl -X POST -F "file=@sample.jpg" http://localhost:8000/api/detect
```

## Model Weights
Place a custom model at `app/models/yolov8n.pt` after training. If missing, the default `yolov8n.pt` will auto-download at first inference.

## Training
Prepare a YOLO dataset YAML (see ultralytics docs). Then run:
```bash
python training/train.py --data path/to/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640
```
After training, the best weights (if found) are copied to `app/models/yolov8n.pt`.

## Deployment (Heroku/Render)
The `Procfile` runs:
```
web: uvicorn app.main:app --host=0.0.0.0 --port=${PORT:-8000}
```
Ensure you set a persistent volume or re-upload weights if the dyno restarts (Heroku ephemeral FS).

## Notes
- Large models may exceed free tier memory.
- Adjust batch size and image size in training for your hardware.
- Add authentication / rate limiting before public exposure.

## License
MIT (adjust if needed).
