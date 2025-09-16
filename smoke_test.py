import os
from app.services.detection import warmup_model, detect_garbage_bytes

# Ensure small input and CPU usage for quick smoke
os.environ.setdefault("YOLO_DEVICE", "cpu")
os.environ.setdefault("YOLO_MAX_SIDE", "320")

# 1. Warmup
warmup_model()

# 2. Create a tiny 10x10 RGB image in memory
from PIL import Image
from io import BytesIO
img = Image.new("RGB", (10, 10), color=(0, 0, 0))
buf = BytesIO()
img.save(buf, format="JPEG")

# 3. Run detection on the blank image, should be fast and likely 0 detections
result = detect_garbage_bytes(buf.getvalue())
print("Smoke test result:", result)
