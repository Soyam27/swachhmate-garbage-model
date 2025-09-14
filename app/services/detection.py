"""Garbage detection service with safe lazy model loading.

The previous version loaded a specific custom weight path at import time:
    runs/detect/train8/weights/best.pt
If that path did not exist, the entire API import failed. This revision:
  * Defers model loading until first inference.
  * Allows overriding path via env var YOLO_MODEL_PATH.
  * Falls back to the default pretrained "yolov8n.pt" if no custom weight found.
  * Raises a clear RuntimeError if ultralytics is missing.
"""

from __future__ import annotations

import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any
from huggingface_hub import hf_hub_download
try:
    from ultralytics import YOLO  # type: ignore
except Exception as e:  # pragma: no cover
    YOLO = None  # type: ignore
    _import_error = e
else:  # pragma: no cover
    _import_error = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

ENV_VAR = "YOLO_MODEL_PATH"

# Runtime tuning environment variables
MAX_SIDE = int(os.getenv("YOLO_MAX_SIDE", "640"))          # Clamp max(width, height)
CONF_THRESH = float(os.getenv("YOLO_CONF", "0.25"))        # Confidence threshold
DEVICE = os.getenv("YOLO_DEVICE", "cpu")                   # e.g. 'cpu', 'cuda:0'
USE_HALF = os.getenv("YOLO_HALF", "false").lower() in {"1", "true", "yes"}

# Attempt to download custom weights once at import time, but don't instantiate model yet.
# If download fails (e.g., no internet), we gracefully fall back later.
try:  # pragma: no cover - network/io
    _downloaded_custom = hf_hub_download(repo_id="avgsoyam/yolo-garbage-detector", filename="best.pt")
except Exception:  # pragma: no cover
    _downloaded_custom = None

# Store only the Path (or None). We previously stored a YOLO model object here which broke .exists() usage.
DEFAULT_CUSTOM: Optional[Path] = Path(_downloaded_custom) if _downloaded_custom else None


class ModelLoadError(RuntimeError):
    pass


def _select_weight_path() -> Optional[Path]:
    """Return a usable weight file path or None for fallback.

    Priority:
      1. Env var YOLO_MODEL_PATH if it exists.
      2. Downloaded custom model (best.pt) if present.
      3. None -> caller will use default pretrained name.
    """
    env_path = os.getenv(ENV_VAR)
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    if DEFAULT_CUSTOM and DEFAULT_CUSTOM.exists():
        return DEFAULT_CUSTOM
    return None


@lru_cache(maxsize=1)
def _get_model():
    if YOLO is None:  # pragma: no cover
        raise ModelLoadError(f"ultralytics import failed: {_import_error}")
    weight = _select_weight_path()
    if weight is not None:
        model = YOLO(str(weight))
    else:
        model = YOLO("yolov8n.pt")
    # Attempt device / half precision adjustment (best effort)
    try:  # pragma: no cover
        model.to(DEVICE)
        if USE_HALF and hasattr(model.model, "half"):
            model.model.half()
    except Exception:
        pass
    return model


def detect_garbage(image_path: str) -> bool:
    """Return True if any detection exists (placeholder logic).

    Customize: filter by class IDs or confidence thresholds if needed.
    """
    model = _get_model()
    results = model(image_path, verbose=False)
    return any(len(r.boxes) > 0 for r in results)


def _maybe_downscale(pil_img):
    if Image is None:
        return pil_img
    w, h = pil_img.size
    m = max(w, h)
    if m <= MAX_SIDE:
        return pil_img
    scale = MAX_SIDE / float(m)
    new_w, new_h = int(w * scale), int(h * scale)
    return pil_img.resize((new_w, new_h))


def detect_garbage_bytes(data: bytes, suffix: str = ".jpg") -> Dict[str, Any]:
    """Run detection on raw bytes, with optional resizing & confidence filter.

    Returns structured result for richer client insight.
    """
    model = _get_model()
    if Image is None:
        raise ModelLoadError("Pillow is required for in-memory image decoding. Install with `pip install Pillow`.")
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise ModelLoadError(f"Invalid image data: {e}")
    img = _maybe_downscale(img)
    # Run inference with configured confidence / device
    results = model(img, verbose=False, conf=CONF_THRESH, device=DEVICE)
    total = 0
    max_conf = 0.0
    for r in results:
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        count = len(boxes)
        total += count
        try:  # attempt to read confidence
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
        "resized": True if max(img.size) <= MAX_SIDE else False
    }
