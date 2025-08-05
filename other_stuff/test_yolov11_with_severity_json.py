import json
import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("/Users/george/Documents/github/final_gemma/gemma3n-car-damage-competition/object_detection/yolov11/train6/weights/best.pt")

# Class names and weights for severity
CLASS_NAMES = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat"
]

CLASS_WEIGHTS = {
    "dent": 2.0,
    "scratch": 2.0,
    "crack": 3.0,
    "glass shatter": 5.0,
    "lamp broken": 3.5,
    "tire flat": 1.0
}

# Run inference
results = model.predict(
    source="./test_images/t7.jpg",
    imgsz=1024,
    conf=0.20,
    iou=0.50,
    save=False,
    verbose=False
)

# Extract image (numpy array)
image = results[0].orig_img.copy()

# Process detections with severity score
detections = []
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    cls_name = CLASS_NAMES[cls_id]
    conf = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0]
    width = float(x2 - x1)
    height = float(y2 - y1)
    area = width * height

    severity_score = CLASS_WEIGHTS[cls_name] * conf * area

    detections.append({
        "class": cls_name,
        "conf": conf,
        "bbox": [x1.item(), y1.item(), x2.item(), y2.item()],
        "area": area,
        "severity_score": severity_score
    })

# Sort by severity
sorted_detections = sorted(detections, key=lambda x: x["severity_score"], reverse=True)

# Print sorted results
print("\n🔧 Ranked Detections by Severity:")
for det in sorted_detections:
    print(
        f"{det['class']} | Conf: {det['conf']:.2f} | Area: {det['area']:.0f} | "
        f"Score: {det['severity_score']:.0f} | BBox: {det['bbox']}"
    )

# Save JSON
with open("ranked_detections.json", "w") as f:
    json.dump(sorted_detections, f, indent=2)

# Draw and save predicted image
def draw_boxes(image, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f"{det['class']} ({int(det['severity_score'])})"
        color = (0, 0, 255)  # Red box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

output_image = draw_boxes(image, sorted_detections)
cv2.imwrite("predicted_output.jpg", output_image)

# Optional: return as dict if you're using this in an API or script
result_json = {
    "image_path": "predicted_output.jpg",
    "detections": sorted_detections
}
