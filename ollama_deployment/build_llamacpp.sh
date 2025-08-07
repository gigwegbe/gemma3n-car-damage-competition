apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggerganov/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DLLAMA_CURL=ON
# cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split llama-mtmd-cli
cmake --build llama.cpp/build --config Release --clean-first --target llama-quantize llama-cli llama-gguf-split llama-mtmd-cli -j4
cp llama.cpp/build/bin/llama-* llama.cpp

# python llama.cpp/convert_hf_to_gguf.py FOLDER --outfile OUTPUT --outtype f16
# python llama.cpp/convert_hf_to_gguf.py gemma-use/ --outfile gguf --outtype f16
# python llama.cpp/convert_hf_to_gguf.py gemma-use/ --outfile gguf --outtype f16 --mmproj . 