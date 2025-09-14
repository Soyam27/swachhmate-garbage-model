from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.detection import detect_garbage_bytes, ModelLoadError
import traceback

router = APIRouter()


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
    finally:
        await file.close()

    try:
        is_garbage = detect_garbage_bytes(content)
    except ModelLoadError as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return {"garbage_detected": is_garbage}
