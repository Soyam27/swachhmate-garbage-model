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
from pathlib import Path
from typing import Optional

try:
    from ultralytics import YOLO  # type: ignore
except Exception as e:  # pragma: no cover
    YOLO = None  # type: ignore
    _import_error = e
else:  # pragma: no cover
    _import_error = None

DEFAULT_CUSTOM = Path("runs/detect/train8/weights/best.pt")
ENV_VAR = "YOLO_MODEL_PATH"


class ModelLoadError(RuntimeError):
    pass


def _select_weight_path() -> Optional[Path]:
    env_path = os.getenv(ENV_VAR)
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    if DEFAULT_CUSTOM.exists():
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
