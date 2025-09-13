from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.detection import detect_garbage, ModelLoadError
from app.utils.preprocess import save_upload
import os
import uuid
import traceback

router = APIRouter()


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    stem, ext = os.path.splitext(file.filename)
    if not ext:
        ext = ".jpg"
    safe_name = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"
    try:
        saved_path = save_upload(file, safe_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    try:
        is_garbage = detect_garbage(saved_path)
    except ModelLoadError as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    return {"garbage_detected": is_garbage, "file": safe_name}
