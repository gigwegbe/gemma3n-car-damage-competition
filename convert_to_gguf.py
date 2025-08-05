import torch
from unsloth import FastLanguageModel

# You need to load your base model and tokenizer.
# Replace this with the actual model you used for fine-tuning.
# For example, a Gemma model.
# The 'max_seq_length' and 'dtype' should match your fine-tuning setup.
model_name = "unsloth/gemma-3n-E2B-it" # Or your base model
max_seq_length = 2048 # Or whatever sequence length you used
dtype = torch.bfloat16 # Or torch.float16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = True, # Use 4-bit loading for faster fine-tuning
)

# Now, you load your fine-tuned model's weights.
# The `model.load_pretrained` function is used for this in Unsloth.
# The directory "mano-wii/gemma-3-finetune" should be a local path
# where your fine-tuned model is saved, or a Hugging Face Hub repository ID.
# It is assumed you have already fine-tuned your model and saved it here.
# For demonstration purposes, we'll assume the fine-tuned model is saved to a local directory.
fine_tuned_model_dir = "gemma-3N-lora-checkpoint"
model = FastLanguageModel.from_pretrained(
    model = model,
    tokenizer = tokenizer,
    lora_path = fine_tuned_model_dir,
)

# Now, use the save_pretrained_gguf function.
# This function first merges the LoRA adapter with the base model weights,
# then converts the merged model to GGUF, and finally applies the quantization.
# You need to provide the tokenizer and the quantization method.
# In your case, it's "Q8_0".
model.save_pretrained_gguf(
    "gguf/gemma-3nn-finetune", # This will be the output directory for the GGUF file
    tokenizer,
    quantization_method = "q8_0"
)

print("GGUF model with Q8_0 quantization saved successfully!")