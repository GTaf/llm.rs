# References
Global architecture : https://medium.com/@hsinhungw/gpt-2-detailed-model-architecture-6b1aad33d16b

Please download GPT-2 weights at : https://huggingface.co/openai-community/gpt2?show_file_info=model.safetensors

# TODO
- [x] Use tiktoken encoder to tranform sentence to tokens list
- [x] Need to find token -> embedding mapping in GPT-2
  - [x] Find how to get the weigts from u8 to f32, then create emebddings
- [x] Find position embedding formula and add matrices together
- [ ] Find how the layer norm works, it's the first layer
- [ ] Create a linear layer pass
