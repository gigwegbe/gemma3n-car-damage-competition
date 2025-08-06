from ultralytics import YOLO

# Load trained model
model = YOLO("yolov11/train6/weights/best.pt")

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
    source="./test_images/t6.jpg",
    imgsz=1024,
    conf=0.20,
    iou=0.50,
    # save_txt=True,
    save=True,
    project="outputs"
    # name="test_infer"           
)

# Visualize
# results[0].show()
# results.save(filename="result.jpg")

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
        "bbox": (x1.item(), y1.item(), x2.item(), y2.item()),
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
