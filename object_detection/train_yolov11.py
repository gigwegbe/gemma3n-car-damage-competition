from ultralytics import YOLO
from ultralytics import settings


# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

print(settings)
results = model.train(
    data="/home/george/Desktop/CarDD-USTC.github.io/code/CarDD_dataset/CarDD_COCO/dataset.yml",
    epochs=500,
    patience=50,  # early stopping
    imgsz=1024,
    batch=16,      # adjust based on VRAM
    augment=True
)
# # Evaluate the model's performance on the validation set
results = model.val()

# # Perform object detection on an image using the model
# results = model("https://ultralytics.com/images/bus.jpg")

# # Export the model to ONNX format
# success = model.export(format="onnx")