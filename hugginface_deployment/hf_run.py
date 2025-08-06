from flask import Flask, request, jsonify
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
import torch
from PIL import Image
import requests
from io import BytesIO
import os 
from werkzeug.utils import secure_filename

# Flask setup
app = Flask(__name__)

# Upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set PyTorch to use only 1 thread (MacBooks are sensitive to CPU use)
torch.set_num_threads(4)

config = AutoConfig.from_pretrained("gigwegbe/gemma3n-merged")
config.num_attention_heads = 8 

# Load model and processor once
processor = AutoProcessor.from_pretrained("gigwegbe/gemma3n-merged")
model = AutoModelForImageTextToText.from_pretrained("gigwegbe/gemma3n-merged", config=config)
device = torch.device("mps")  # Force CPU for MacBook
model.eval()

def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('file')
    question = request.form.get('question')
    if file is None or question is None:
        return jsonify({'error': 'Image file and question are required'}), 400
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file or extension'}), 400

    fname = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(path)

    image = Image.open(path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]
    }]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=40)

    answer = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return jsonify({'answer': answer})



@app.route('/')
def home():
    return "Gemma3n Visual QA API running on MacBook."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6000, use_reloader=False)



#================= Send this via Terminal =================
# curl -X POST http://localhost:6000/analyze \
#   -F "file=@/Users/george/Desktop/car2.png" \
#   -F "question=What damages done to the car?"
#================= Send this via Terminal =================