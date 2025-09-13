from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train8/weights/best.pt")

# Folder containing images you want to test
image_paths = [
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image1.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image2.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image3.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image4.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image5.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image6.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image7.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image8.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image9.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image10.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image11.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image12.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image13.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image14.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image15.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image16.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image17.jpg",
    "C:\\Users\\soyam\\OneDrive\\Desktop\\Yolo-Model\\test_imgs\\image18.jpg",
]

# Run inference on each image and save to a custom folder
for img_path in image_paths:
    results = model(img_path, save=True, save_dir="runs/detect/inference_results")  # save annotated images
    print(f"Processed {img_path}, results saved to runs/detect/inference_results")
