from ultralytics import YOLO
from ultralytics import settings

# Create a new YOLO model from scratch
model = YOLO("yolov8n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.pt")

print(settings)
# # Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="/home/george/Desktop/CarDD-USTC.github.io/code/CarDD_dataset/CarDD_COCO/dataset.yml", epochs=1000, augment=True)

# # Evaluate the model's performance on the validation set
results = model.val()

# # Perform object detection on an image using the model
# results = model("https://ultralytics.com/images/bus.jpg")

# # Export the model to ONNX format
# success = model.export(format="onnx")