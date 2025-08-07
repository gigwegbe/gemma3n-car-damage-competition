# AutoVision Inspector
![](./assets/webapp-results/m4.png)

Dataset:
- Website - [Link](https://cardd-ustc.github.io/)
- Paper - [Link](https://cardd-ustc.github.io/docs/CarDD.pdf)
- Dataset - [Link](https://drive.google.com/file/d/1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx/view)
- Our Annotated Dataset - [Link](https://huggingface.co/datasets/gigwegbe/damaged-car-dataset-annotated)

Model: 
- Finetune Adapter - [Link](https://huggingface.co/gigwegbe/gemma-3n-E2B-it-finetuned-adapters)
- Merge Model(Based Model and Finetunned Adapter) - [Link](https://huggingface.co/gigwegbe/gemma3n-merged)
- GGUF Models - [Link](https://huggingface.co/gigwegbe/gemma3n-gguf)
- Training Logs (wandb) - [Link](https://wandb.ai/gigwegbe-carnegie-mellon-university/my-vision-finetune?nw=nwusergigwegbe)
  


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


## How to Run the Application
Start the AutoVision Inspector VLM
- Launch VLM Deployment - [Link](https://github.com/gigwegbe/gemma3n-car-damage-competition/tree/main/hugginface_deployment)

Start the WebSocket Server and Frontend
- Start WebSocket & Serve Frontend - [Link](https://github.com/gigwegbe/gemma3n-car-damage-competition/tree/main/object_detection)



### TODO 
- Add wandb log - Done 
- While merging, monitor GPU memory usage with (watch -n 1 nvidia-smi) - Done 
- Save image locally - Done( Kindly review)
  - Test the system with new JSON format - Done 
  - Review prompt("Only include entries that are visible in the image.") - Done 
- Inference Script - Done 
- Add frontend material - Done 
- Add Backend material - Done 
- Add Report 
- Upload Video 
- Remove all token or make them invalid - Done 
- Add Notebook to Kaggle
- Add paper - Done 
- Add Design - GoodNote - Done 
- Add Wandb 
  - Link - Done
  - Images - Done 
- Update Huggingface readme 
- Add Readme of the following:
  - training - Done 
  - inference - Done 
  - deployment - Done 
  - notebooks - Done 
  - Ollama deployment - ?
  - Inference reference - ?
  - local huggingface deployment
      - review the `max_new_tokens` currently 40 change to smt - Done 
      - review the attention head ? 8 - Done 
      - Macbook M4 - Done 
      - RTX 3090 - Done 
      - Add requirements for deployment - Done 

- Review all links are working 
  


  # References
  - 
  - [Running Ollama 2 on NVIDIA Jetson Nano with GPU using Docker](https://collabnix.com/running-ollama-2-on-nvidia-jetson-nano-with-gpu-using-docker/)
