import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# 1. Set device to MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load processor and model
processor = AutoProcessor.from_pretrained("gigwegbe/gemma3n-merged")
model = AutoModelForImageTextToText.from_pretrained("gigwegbe/gemma3n-merged").to(device)

# 3. Prepare the messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]

# 4. Tokenize input and move to device
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 5. Generate and decode output
outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
