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
from typing import Optional
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
        return YOLO(str(weight))
    # Fallback to base pretrained weights (downloaded if missing)
    return YOLO("yolov8n.pt")


def detect_garbage(image_path: str) -> bool:
    """Return True if any detection exists (placeholder logic).

    Customize: filter by class IDs or confidence thresholds if needed.
    """
    model = _get_model()
    results = model(image_path, verbose=False)
    return any(len(r.boxes) > 0 for r in results)


def detect_garbage_bytes(data: bytes, suffix: str = ".jpg") -> bool:
    """Run detection directly on raw image bytes without persisting to uploads.

    Ultralytics YOLO predict() can take a BytesIO stream or numpy/PIL image. We
    feed a BytesIO to avoid disk writes. Suffix hints the image format.
    """
    model = _get_model()
    if Image is None:
        raise ModelLoadError("Pillow is required for in-memory image decoding. Install with `pip install Pillow`.")
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise ModelLoadError(f"Invalid image data: {e}")
    results = model(img, verbose=False)
    return any(len(r.boxes) > 0 for r in results)
