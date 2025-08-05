sudo apt update && sudo apt upgrade -y
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version # 25.5.1 

# Conda 
conda create -n myenv python=3.11
conda activate myenv

# Training setup. 
pip install unsloth
pip install --no-deps transformers==4.53.1 # Only for Gemma 3N
pip install --no-deps --upgrade timm # Only for Gemma 3N
pip install wandb 

ollama create gigwegbe-gemma3n-merged -f ./gigwegbe-gemma3n-merged