from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch

import torch._dynamo
torch._dynamo.config.cache_size_limit = 1024 

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, processor = FastVisionModel.from_pretrained(
    "unsloth/gemma-3n-E2B",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)



model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 32,                           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 32,                  # Recommended alpha == r at least
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
dataset = load_dataset("unsloth/LaTeX_OCR", split = "train")


from IPython.display import display, Math, Latex

latex = dataset[3]["text"]


instruction = "Write the LaTeX representation for this image."

def convert_to_conversation(sample):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
    ]
    return {"messages": conversation}
pass



converted_dataset = [convert_to_conversation(sample) for sample in dataset]
converted_dataset[0]


from unsloth import get_chat_template
processor = get_chat_template(
    processor,
    "gemma-3n"
)

"""Before fine-tuning, let us evaluate the base model's performance. We do not expect strong results, as it has not encountered this chat template before."""

# FastVisionModel.for_inference(model)  # Enable for inference!

# image = dataset[2]["image"]
# instruction = "Write the LaTeX representation for this image."

# messages = [
#     {
#         "role": "user",
#         "content": [{"type": "image"}, {"type": "text", "text": instruction}],
#     }
# ]
# input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
# inputs = processor(
#     image,
#     input_text,
#     add_special_tokens=False,
#     return_tensors="pt",
# ).to("cuda")

# from transformers import TextStreamer

# text_streamer = TextStreamer(processor, skip_prompt=True)
# result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
#                         use_cache=True, temperature = 1.0, top_p = 0.95, top_k = 64)

from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model=model,
    train_dataset=converted_dataset,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        gradient_checkpointing = True,

        # use reentrant checkpointing
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        max_grad_norm = 0.3,              # max gradient norm based on QLoRA paper
        warmup_ratio = 0.03,
        max_steps = 2,
        #num_train_epochs = 2,          # Set this instead of max_steps for full training runs
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
        max_length = 2048,
    )
)

# @title Show current memory stats
# gpu_stats = torch.cuda.get_device_properties(0)
# start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
# print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
# print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# # @title Show final memory and time stats
# used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
# used_percentage = round(used_memory / max_memory * 100, 3)
# lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
# print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
# print(
#     f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
# )
# print(f"Peak reserved memory = {used_memory} GB.")
# print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
# print(f"Peak reserved memory % of max memory = {used_percentage} %.")
# print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

FastVisionModel.for_inference(model)  # Enable for inference!

# image = dataset[10]["image"]
# instruction = "Write the LaTeX representation for this image."

# messages = [
#     {
#         "role": "user",
#         "content": [{"type": "image"}, {"type": "text", "text": instruction}],
#     }
# ]

# input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

# inputs = processor(
#     image,
#     input_text,
#     add_special_tokens=False,
#     return_tensors="pt",
# ).to("cuda")

# from transformers import TextStreamer

# text_streamer = TextStreamer(processor, skip_prompt=True)
# result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
#                         use_cache=True, temperature = 1.0, top_p = 0.95, top_k = 64)


model.save_pretrained("lora_model")  # Local saving
processor.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# processor.push_to_hub("your_name/lora_model", token = "...") # Online saving


if False:
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name="lora_model",  # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit=True,  # Set to False for 16bit LoRA
    )
    FastVisionModel.for_inference(model)  # Enable for inference!

FastVisionModel.for_inference(model)  # Enable for inference!

# sample = dataset[1]
# image = sample["image"].convert("RGB")
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "text",
#                 "text": sample["text"],
#             },
#             {
#                 "type": "image",
#             },
#         ],
#     },
# ]
# input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
# inputs = processor(
#     image,
#     input_text,
#     add_special_tokens=False,
#     return_tensors="pt",
# ).to("cuda")

# from transformers import TextStreamer

# text_streamer = TextStreamer(processor.tokenizer, skip_prompt=True)
# _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
#                    use_cache=True, temperature = 1.0, top_p = 0.95, top_k = 64)

# """### Saving to float16 for VLLM

# We also support saving to `float16` directly. Select `merged_16bit` for float16. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.
# """

# Select ONLY 1 to save! (Both not needed!)

# Save locally to 16bit
if True: model.save_pretrained_merged("unsloth_finetune", processor,)

# To export and save to your Hugging Face account
# if False: model.push_to_hub_merged("YOUR_USERNAME/unsloth_finetune", processor, token = "PUT_HERE")


model.save_pretrained_gguf(
    "unsloth_finetune", # The output directory for the GGUF file
    quantization_method = "f16" # Correct keyword argument name
)

# if True: model.save_pretrained_gguf("model", processor, quantization_method = "f16")