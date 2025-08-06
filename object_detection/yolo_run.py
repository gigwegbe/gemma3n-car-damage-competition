from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import os
import tempfile
import uuid
from PIL import Image
import base64
import io
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('outputs', exist_ok=True)

try:
    model = YOLO("yolov11/train6/weights/best.pt")
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None

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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_detections(results):
    """Process YOLO results into sorted detections with severity scores"""
    detections = []
    
    if len(results) == 0 or results[0].boxes is None:
        return detections
        
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
    return sorted_detections

def format_detections_string(sorted_detections):
    """Format detections as a string with the exact formatting style"""
    if not sorted_detections:
        return "No detections found."
    
    formatted_lines = ["Ranked Detections by Severity:"]
    
    for det in sorted_detections:
        line = (f"{det['class']} | Conf: {det['conf']:.2f} | Area: {det['area']:.0f} | "
                f"Score: {det['severity_score']:.0f} | BBox: {det['bbox']}")
        formatted_lines.append(line)
    
    return "\n".join(formatted_lines)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "YOLO Car Damage Detection API",
        "endpoints": {
            "/detect": "POST - Upload image for damage detection (returns JSON with formatted text)",
            "/detect_text": "POST - Upload image for damage detection (returns formatted text only)",
            "/detect_file_only": "POST - Upload image for damage detection (returns processed image file)",
            "/health": "GET - Check API health"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    })

@app.route('/detect', methods=['POST'])
def detect_damage():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Use: png, jpg, jpeg, gif, bmp, tiff"}), 400

    try:
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{unique_id}.{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(filepath)

        conf_threshold = float(request.form.get('conf', 0.20))
        iou_threshold = float(request.form.get('iou', 0.50))
        img_size = int(request.form.get('imgsz', 1024))

        results = model.predict(
            source=filepath,
            imgsz=img_size,
            conf=conf_threshold,
            iou=iou_threshold,
            save=True,
            project="outputs",
            name=unique_id,
            exist_ok=True
        )

        sorted_detections = process_detections(results)

        formatted_detections = format_detections_string(sorted_detections)
        output_image_path = None
        output_dir = f"outputs/{unique_id}"
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    output_image_path = os.path.join(output_dir, file)
                    break

        response_data = {
            "detections": sorted_detections,
            "formatted_detections": formatted_detections,
            "detection_count": len(sorted_detections),
            "parameters": {
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "image_size": img_size
            }
        }

        # with open("temp.txt", "w") as temp_file:
        #     temp_file.write(formatted_detections)
        # with open("temp_w.txt", "w") as temp_w_file:
        #     temp_w_file.write(json.dumps(response_data, indent=2))
        if output_image_path and os.path.exists(output_image_path):
            with open(output_image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()
                # response_data["output_image"] = f"data:image/{file_ext};base64,{img_base64}"
                response_data["output_image_path"] = output_image_path
        if os.path.exists(filepath):
            os.remove(filepath)
        # with open("temp_finale.txt", "w") as temp_finale_file:
        #     temp_finale_file.write(json.dumps(response_data, indent=2))
        return jsonify(response_data)

    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/detect_text', methods=['POST'])
def detect_damage_text():
    """Endpoint that returns only the formatted text results"""
    if model is None:
        return "Error: Model not loaded", 500

    if 'image' not in request.files:
        return "Error: No image file provided", 400

    file = request.files['image']
    
    if file.filename == '':
        return "Error: No file selected", 400

    if not allowed_file(file.filename):
        return "Error: File type not allowed. Use: png, jpg, jpeg, gif, bmp, tiff", 400

    try:
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{unique_id}.{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        conf_threshold = float(request.form.get('conf', 0.20))
        iou_threshold = float(request.form.get('iou', 0.50))
        img_size = int(request.form.get('imgsz', 1024))
        results = model.predict(
            source=filepath,
            imgsz=img_size,
            conf=conf_threshold,
            iou=iou_threshold,
            save=False, 
            project="outputs",
            name=unique_id,
            exist_ok=True
        )
        sorted_detections = process_detections(results)
        formatted_detections = format_detections_string(sorted_detections)
        if os.path.exists(filepath):
            os.remove(filepath)
        return formatted_detections, 200, {'Content-Type': 'text/plain; charset=utf-8'}

    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return f"Error: Processing failed: {str(e)}", 500

@app.route('/detect_file_only', methods=['POST'])
def detect_damage_file_only():
    """Alternative endpoint that returns the processed image file directly"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    try:
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{unique_id}.{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        conf_threshold = float(request.form.get('conf', 0.20))
        iou_threshold = float(request.form.get('iou', 0.50))
        img_size = int(request.form.get('imgsz', 1024))
        results = model.predict(
            source=filepath,
            imgsz=img_size,
            conf=conf_threshold,
            iou=iou_threshold,
            save=True,
            project="outputs",
            name=unique_id,
            exist_ok=True
        )

        output_image_path = None
        output_dir = f"outputs/{unique_id}"
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    output_image_path = os.path.join(output_dir, file)
                    break

        if os.path.exists(filepath):
            os.remove(filepath)

        if output_image_path and os.path.exists(output_image_path):
            return send_file(output_image_path, as_attachment=True, download_name=f"detected_{filename}")
        else:
            return jsonify({"error": "No output image generated"}), 500

    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

if __name__ == '__main__':
    print(" Starting YOLO Car Damage Detection API...")
    print(" Available endpoints:")
    print("  GET  /          - API information")
    print("  GET  /health    - Health check")
    print("  POST /detect    - Detect damage (returns JSON with formatted text)")
    print("  POST /detect_text - Detect damage (returns formatted text only)")
    print("  POST /detect_file_only - Detect damage (returns processed image file)")
    print("\nUsage examples:")
    print("  curl -X POST -F 'image=@your_image.jpg' http://localhost:5000/detect")
    print("  curl -X POST -F 'image=@your_image.jpg' http://localhost:5000/detect_text")
    print("  curl -X POST -F 'image=@your_image.jpg' -F 'conf=0.3' -F 'iou=0.5' http://localhost:5000/detect_text")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
