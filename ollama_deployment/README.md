There are different way to deploy to Ollama: 
We tried 2 methods: 

- Using SafeTensor model from Huggingface
- Pulling Model from Hugging
- Using GGUF model  converted using Llamacpp, check this for more details[](): 
    To building llamacpp: 
    ```
    ./build_gguf_llamacpp.sh
    ```
    After that run the command below to convert the model: 
    ```
    python llama.cpp/convert_hf_to_gguf.py gemma-use/ --outfile gguf --outtype f16
    ```

    To build the gguf model with the vision projection adapter
    ```

python llama.cpp/convert_hf_to_gguf.py gemma-use/ --outfile gguf --outtype f16 --mmproj . 
    ```


Quantization of Models: 
We converted the model to various formats to support  various low powered devices: 
<Add Image>

Link to the  Models(GGUF) repository - Link 

Deployment on different device: 

- Macbook M4: 
   <Add Image>


- Jetson TX2 
    <Add Image>


- Jetson Nano 4GB
    <Add Image>