from ultralytics import YOLO

# Load your fine-tuned YOLO model (replace with your path)
model = YOLO("/Users/george/Documents/github/final_gemma/gemma3n-car-damage-competition/object_detection/yolov8/train3/weights/best.pt")

# Perform inference on a single image
results = model.predict(source="./test_images/t1.jpg", imgsz=1024, conf=0.20, iou=0.50)
# Visualize or display detection results
results[0].show()

# Access bounding boxes, labels, and confidences
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0]  # bounding box coordinates
    print(f"Class {cls_id}, Conf: {confidence:.2f}, BBox: {x1},{y1},{x2},{y2}")
