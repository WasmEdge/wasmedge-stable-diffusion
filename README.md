# WasmEdge Stable Diffusion
A Rust library for using stable diffusion functions when the Wasi is being executed on WasmEdge.

## Set up WasmEdge
```
git clone https://github.com/WasmEdge/WasmEdge.git
cd WasmEdge
cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release -DWASMEDGE_BUILD_TESTS=OFF -DWASMEDGE_PLUGIN_STABLEDIFFUSION=On
cmake --build build
cmake --install build
```

## Download Model
Download the weights or quantized model from the following command.  
You also can use our example to quantize the weights by yourself.

stable-diffusion v1.4: [second-state/stable-diffusion-v-1-4-GGUF](https://huggingface.co/second-state/stable-diffusion-v-1-4-GGUF)  
stable-diffusion v1.5: [second-state/stable-diffusion-v1-5-GGUF](https://huggingface.co/second-state/stable-diffusion-v1-5-GGUF)  
stable-diffusion v2.1: [second-state/stable-diffusion-2-1-GGUF](https://huggingface.co/second-state/stable-diffusion-2-1-GGUF)

```
curl -L -O https://huggingface.co/second-state/stable-diffusion-v-1-4-GGUF/resolve/main/sd-v1-4.ckpt

curl -L -O https://huggingface.co/second-state/stable-diffusion-v-1-4-GGUF/resolve/main/stable-diffusion-v1-4-Q8_0.gguf
```

## Run the example
```
cargo build --target wasm32-wasi --release
wasmedge --dir .:. ./target/wasm32-wasi/release/wasmedge_stable_diffusion_example.wasm
```

Then you can see the three new files.
1. sd-v1-4-Q8_0.gguf: is the quantization of sd-v1-4
2. output.png: an image with a cat
3. output2.png: an image of a cat with blue eyes.