"""
Optimized Garbage Detection Service:

- Prefers a local cached model for fast startup.
- Optional first-start download from Hugging Face Hub.
- Warm-up model on app startup to avoid first-request delay.
- Falls back to ``yolov8n.pt`` if no custom model is found.
"""

from __future__ import annotations
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    _import_error = e
else:
    _import_error = None

try:
    from PIL import Image
except Exception:
    Image = None

# Environment & runtime configs
ENV_VAR = "YOLO_MODEL_PATH"
MAX_SIDE = int(os.getenv("YOLO_MAX_SIDE", "640"))
CONF_THRESH = float(os.getenv("YOLO_CONF", "0.25"))
DEVICE = os.getenv("YOLO_DEVICE", "cpu")
USE_HALF = os.getenv("YOLO_HALF", "false").lower() in {"1", "true", "yes"}
CPU_THREADS = os.getenv("YOLO_CPU_THREADS")

# Local model directory relative to this file (absolute path)
MODEL_DIR = (Path(__file__).resolve().parents[1] / "model").resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Pre-download from Hugging Face if not exists
CUSTOM_MODEL_PATH = MODEL_DIR / "best.pt"
if not CUSTOM_MODEL_PATH.exists():
    try:
        from huggingface_hub import hf_hub_download
        CUSTOM_MODEL_PATH = Path(
            hf_hub_download(
                repo_id="avgsoyam/yolo-garbage-detector",
                filename="best.pt",
                cache_dir=str(MODEL_DIR)
            )
        )
    except Exception:
        CUSTOM_MODEL_PATH = None

# Use ENV_VAR path first
def _select_weight_path() -> Optional[Path]:
    env_path = os.getenv(ENV_VAR)
    if env_path and Path(env_path).exists():
        return Path(env_path)
    if CUSTOM_MODEL_PATH and CUSTOM_MODEL_PATH.exists():
        return CUSTOM_MODEL_PATH
    return None

class ModelLoadError(RuntimeError):
    pass

@lru_cache(maxsize=1)
def _get_model():
    if YOLO is None:
        raise ModelLoadError(f"ultralytics import failed: {_import_error}")
    weight = _select_weight_path()
    if weight is not None:
        model = YOLO(str(weight))
    else:
        model = YOLO("yolov8n.pt")
    try:
        # Optional: tune CPU threads to prevent oversubscription
        if CPU_THREADS:
            try:
                import torch  # type: ignore
                torch.set_num_threads(int(CPU_THREADS))
            except Exception:
                pass
        model.to(DEVICE)
        if USE_HALF and hasattr(model.model, "half"):
            model.model.half()
    except Exception:
        pass
    return model

# Warm-up model at startup (call this in FastAPI/Flask startup event)
def warmup_model():
    try:
        _get_model()
    except Exception as e:
        print(f"Warning: model warm-up failed: {e}")

def detect_garbage(image_path: str) -> bool:
    model = _get_model()
    results = model(image_path, verbose=False, conf=CONF_THRESH)
    return any(len(r.boxes) > 0 for r in results)

def _maybe_downscale(pil_img):
    if Image is None:
        return pil_img
    w, h = pil_img.size
    m = max(w, h)
    if m <= MAX_SIDE:
        return pil_img
    scale = MAX_SIDE / float(m)
    return pil_img.resize((int(w*scale), int(h*scale)))

def detect_garbage_bytes(data: bytes, suffix: str = ".jpg") -> Dict[str, Any]:
    model = _get_model()
    if Image is None:
        raise ModelLoadError("Pillow is required for in-memory image decoding.")
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise ModelLoadError(f"Invalid image data: {e}")
    # Determine if we will resize before actually doing it
    orig_w, orig_h = img.size
    resized_flag = max(orig_w, orig_h) > MAX_SIDE
    img = _maybe_downscale(img)
    # device already set on the model; do not pass device each call
    results = model(img, verbose=False, conf=CONF_THRESH)
    total = 0
    max_conf = 0.0
    for r in results:
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        count = len(boxes)
        total += count
        try:
            if count and hasattr(boxes, 'conf'):
                cmax = float(boxes.conf.max().item())
                if cmax > max_conf:
                    max_conf = cmax
        except Exception:
            pass
    return {
        "garbage_detected": total > 0,
        "detections": total,
        "max_confidence": round(max_conf, 4),
        "confidence_threshold": CONF_THRESH,
        "resized": resized_flag
    }
