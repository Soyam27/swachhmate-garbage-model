from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.detection import detect_garbage_bytes, ModelLoadError
import traceback
import os

MAX_UPLOAD_BYTES = int(os.getenv("YOLO_MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))  # 5MB default

router = APIRouter()


@router.get("/health")
async def health():
    """Lightweight health check that does NOT trigger model load."""
    return {"status": "ok"}


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large. Limit {MAX_UPLOAD_BYTES} bytes")
    finally:
        await file.close()

    try:
        result = detect_garbage_bytes(content)
    except ModelLoadError as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return result
