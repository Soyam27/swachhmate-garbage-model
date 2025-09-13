import os
from ultralytics import YOLO
if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    model = YOLO("yolov8n.pt")
    model.train(
        data="C:/Users/soyam/OneDrive/Desktop/Yolo-Model/datasets/garbage/data.yaml",
        epochs=10,
        imgsz=640,
        batch=16
    )
