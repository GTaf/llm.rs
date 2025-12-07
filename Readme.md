# References
Global architecture : https://medium.com/@hsinhungw/gpt-2-detailed-model-architecture-6b1aad33d16b

Please download GPT-2 weights at : https://huggingface.co/openai-community/gpt2?show_file_info=model.safetensors

## CLI Use

Run with `cargo run --release`

## Web use
Build with `wasm-pack build --target web`

Run website with `python3 -m http.server 8000`

For some bench : `cargo run --release --bin bench-gemm`
Fo running local model only : `cargo run --release -- --use-gpu` 

# TODO
- [x] Use tiktoken encoder to tranform sentence to tokens list
- [x] Need to find token -> embedding mapping in GPT-2
  - [x] Find how to get the weigts from u8 to f32, then create emebddings
- [x] Find position embedding formula and add matrices together
- [x] Find how the layer norm works, it's the first layer
- [x] Create a linear layer pass
