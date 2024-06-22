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
```
curl -L -O https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
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