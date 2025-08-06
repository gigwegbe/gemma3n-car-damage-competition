from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch

model, processor = FastVisionModel.from_pretrained(
    "unsloth/gemma-3n-E2B-it",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,                           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,                  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,               # We support rank stabilized LoRA
    loftq_config = None,               # And LoftQ
    target_modules = "all-linear",    # Optional now! Can specify a list if needed
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)



from datasets import load_dataset
data = load_dataset("gigwegbe/damaged-car-dataset-annotated", split="train")

dataset = data 


instruction = (
    "You are an expert automobile inspector. "
    "Describe the visible car damages in the image, "
    "mentioning damage types and approximate regions."
)

def convert_to_conversation(sample):
    # sample["category_id"] is assumed to be a string or list of strings like "dent", "scratch"
    labels = sample.get("category_id", [])
    if isinstance(labels, str):
        unique_labels = [labels]
    else:
        # deduplicate preserving order
        unique_labels = []
        for lbl in labels:
            if lbl not in unique_labels:
                unique_labels.append(lbl)

    answer_text = (
        "There is " + ", ".join(unique_labels) + "." if unique_labels else "No visible damage."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer_text}
            ],
        },
    ]
    return {"messages": messages}



converted_dataset = [convert_to_conversation(sample) for sample in dataset]


import torch._dynamo.config
torch._dynamo.config.cache_size_limit = 100


from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model=model,
    train_dataset=converted_dataset,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor, resize=512),
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        gradient_checkpointing = False,

        # use reentrant checkpointing
        # gradient_checkpointing_kwargs = {"use_reentrant": False},
        max_grad_norm = 0.3,              # max gradient norm based on QLoRA paper
        warmup_steps = 5,                 # Use when using max_steps
        max_steps = 60,
        # warmup_ratio = 0.03,
        # num_train_epochs = 2,           # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        save_strategy="steps",
        optim = "adamw_torch_fused",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",             # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        # max_seq_length = 2048,
    )
)



trainer_stats = trainer.train()

tokenizer = processor.tokenizer
model.save_pretrained("gemma-3N-lora-checkpoint", maximum_memory_usage = 0.75)
tokenizer.save_pretrained("gemma-3N-lora-checkpoint")


# You must use a new folder name since push_to_hub() is for adapters only
model.push_to_hub(
    "gigwegbe/gemma-3n-E2B-it-finetuned-adapters",
    tokenizer=tokenizer,
    token="hf_LOLOL"
)

print("saving pretrained model")
model.save_pretrained_merged(
    "gemma-3N-lora-checkpoint",
    save_method = "merged_4bit_forced"
)

print("pushing  pretrained model")
# Push the merged model to Hugging Face
# Replace "your_username/your_model_name" with your actual Hugging Face username and the desired model name
model.push_to_hub_merged(
    "gigwegbe/gemma-3n-E2B-it-finetuned-4bit",
    tokenizer = tokenizer,
    save_method = "merged_4bit_forced",
    token ="hf_LOLOL"
)