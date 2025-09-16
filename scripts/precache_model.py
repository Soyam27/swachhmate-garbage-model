"""
Optional pre-cache script for deployment builds.
Downloads a preferred model weight so the first request doesn't pay the download cost.
Will skip silently on failure.
"""
from pathlib import Path
import os

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

MODEL_DIR = (Path(__file__).resolve().parents[1] / "app" / "model").resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEST = MODEL_DIR / "best.pt"

if DEST.exists():
    print("Model already present:", DEST)
else:
    print("Attempting to pre-cache model to:", DEST)
    if hf_hub_download is None:
        print("huggingface_hub not available; skipping.")
    else:
        try:
            path = hf_hub_download(
                repo_id=os.getenv("HF_REPO", "avgsoyam/yolo-garbage-detector"),
                filename=os.getenv("HF_FILE", "best.pt"),
                cache_dir=str(MODEL_DIR)
            )
            print("Downloaded to cache:", path)
        except Exception as e:
            print("Pre-cache failed:", e)
