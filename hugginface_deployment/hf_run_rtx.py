from flask import Flask, request, jsonify
from unsloth import FastVisionModel
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
import torch
from PIL import Image
import requests
from io import BytesIO
import os 
from werkzeug.utils import secure_filename
 # FastLanguageModel for LLMs
from transformers import AutoTokenizer, AutoModelForVision2Seq, TextStreamer, AutoProcessor
import torch


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

model, processor = FastVisionModel.from_pretrained(
    "unsloth/gemma-3n-E2B-it",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)
model.load_adapter("gigwegbe/gemma-3n-E2B-it-finetuned-adapters")

# Step 3: Switch to inference mode
FastVisionModel.for_inference(model)

CLASS_WEIGHTS = {
    "dent": 2.0,
    "scratch": 2.0,
    "crack": 3.0,
    "glass shatter": 5.0,
    "lamp broken": 3.5,
    "tire flat": 1.0
}
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
    instruction = f"""You are an expert automobile inspector.
        Review the image and verify whether the car damages shown are consistent with the following object detection results (ranked by severity):

        {question} Recall: {CLASS_WEIGHTS}

        For the detected damages, in a combined fashion return your analysis in the following structured format:


        {{
            "general_description": <Long detailed assessment of all the  damages,  location and severity>,
            "damage_type": <types of damages observed, e.g., 'glass shatter'>,
            "qualitative_description_of_size_of_damage": <for each of the items discovered e.g., 'large', 'medium', 'minor'>,
            "technical_description_location_of_damage": <specific parts of the car affected, e.g., 'front windshield, lower right'>,
            "recommendation_for_fix": < Recommendation for fix for each part e.g., 'Replace the windshield', 'Polish the surface'>,
            "estimated_time_of_repair": <e.g., '2–3 hours of labor', '30 minutes for polishing'>
        }}

        Only include entries that are visible in the image. Be as precise and professional as possible.
        """
        
    messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction}
        ],
    }
    ]

    # This returns a string formatted for chat prompting
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    # === Tokenize multimodal input ===
    inputs = processor(
        text=input_text,
        images=image,
        return_tensors="pt"
    ).to("cuda") 
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        use_cache=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )

    # === Decode only the generated tokens (excluding input) ===
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
    print(generated_text)

    return jsonify({'answer': generated_text})



@app.route('/')
def home():
    return "Gemma3n Visual QA API running on MacBook."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6000, use_reloader=True)



#================= Send this via Terminal =================
# curl -X POST http://localhost:6000/analyze \
#   -F "file=@/Users/george/Desktop/car2.png" \
#   -F "question=What damages done to the car?"
#================= Send this via Terminal =================