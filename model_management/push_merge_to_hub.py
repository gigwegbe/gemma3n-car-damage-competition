from huggingface_hub import HfApi, upload_folder

hf_token = "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"  # Your Hugging Face token
repo_id = "gigwegbe/gemma3n-merged"  # Replace with your desired repo

# Step 1: Create the repo if it doesn't exist
api = HfApi()
api.create_repo(
    repo_id=repo_id,
    token=hf_token,
    repo_type="model",
    exist_ok=True,
    private=False
)

# Step 2: Upload the folder
upload_folder(
    folder_path="gemma-use",
    repo_id=repo_id,
    repo_type="model",
    token=hf_token,
)
