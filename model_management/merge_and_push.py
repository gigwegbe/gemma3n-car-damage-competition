from unsloth import FastVisionModel
import torch
import gc

# Optional: free up CUDA cache and run GC just in case
torch.cuda.empty_cache()
gc.collect()

# Load model + processor from LoRA checkpoint
model, processor = FastVisionModel.from_pretrained(
    "gemma-3N-lora-checkpoint",
    load_in_4bit = True,
    torch_dtype = torch.float16,  # Optional, reduces memory
)

# Save merged model to local folder (4-bit quantized)
print("Saving merged 4-bit model...")
model.save_pretrained_merged(
    "gemma-3N-lora-checkpoint-merged",
    save_method = "merged_4bit_forced",  # Required for 4-bit + vision
)

# Save tokenizer/processor
processor.tokenizer.save_pretrained("gemma-3N-lora-checkpoint-merged")

# Optional: Push to Hugging Face Hub
print("Pushing merged model to Hugging Face Hub...")
model.push_to_hub_merged(
    repo_id = "gigwegbe/gemma-3n-E2B-it-finetuned-4bit",
    tokenizer = processor.tokenizer,
    save_method = "merged_4bit_forced",
    token = "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",  # Replace with your token securely
)

print("✅ Merged model saved and pushed successfully.")
