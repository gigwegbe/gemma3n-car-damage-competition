# gemma3n-car-damage-competition

Dataset:
- Website - [Link](https://cardd-ustc.github.io/)
- Paper - [Link](https://cardd-ustc.github.io/docs/CarDD.pdf)
- Dataset - [Link](https://drive.google.com/file/d/1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx/view)
- Our Annotated Dataset - [Link](https://huggingface.co/datasets/gigwegbe/damaged-car-dataset-annotated)

Model: 
- Finetune Adapter - [Link](https://huggingface.co/gigwegbe/gemma-3n-E2B-it-finetuned-adapters)
- Merge Model(Based Model and Finetunned Adapter) - [Link](https://huggingface.co/gigwegbe/gemma3n-merged)
- GGUF Models - [Link](https://huggingface.co/gigwegbe/gemma3n-gguf)
  


## Flow - Initial Trial 
- Train 
- Push the merge model to hub (Optional)
- Convert the merged model to gguf (Might have to create a swap memory)
- Convert to different format()
- Run on Ollama 


## Flow - Initial Trial 
- Train 
- Push the merge model to hub (Optional)
- Deploy 


### TODO 
- Add wandb log - Done 
- While merging, monitor GPU memory usage with (watch -n 1 nvidia-smi) - Done 
- Save image locally - Done( Kindly review)
  - Test the system with new JSON format
  - Review prompt("Only include entries that are visible in the image.")
- Inference reference
- Inference Script 
- Add frontend material 
- Add Backend meterial 
- Add Report 
- Upload Video 
- Remove all token or make them invalid 
- Add Notebook to Kaggle
- Add paper - Done 
- Add Design - GoodNote - Done 
- Add Wandb 
  - [Link](https://wandb.ai/gigwegbe-carnegie-mellon-university/my-vision-finetune?nw=nwusergigwegbe) - Done
  - Images - Done 
- Read Huggingface readme 
  


  # References
  - 
  - [Running Ollama 2 on NVIDIA Jetson Nano with GPU using Docker](https://collabnix.com/running-ollama-2-on-nvidia-jetson-nano-with-gpu-using-docker/)
