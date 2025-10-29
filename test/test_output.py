import json
from transformers import GPT2Model, GPT2Tokenizer
import torch

# Load model and tokenizer
model = GPT2Model.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Input text
text = "The main character of The lord of the rings is "
input_ids = tokenizer.encode(text, return_tensors="pt")
print(input_ids[0][0])

# Get embeddings (output of embedding layer)
model.eval()
with torch.no_grad():
    saving_dict = dict()
    # Token embeddings
    token_embeddings = model.wte(input_ids)
    saving_dict["Token embeddings"] = token_embeddings[0][0].tolist()

    # Positional embeddings
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length).unsqueeze(0)
    position_embeddings = model.wpe(position_ids)
    saving_dict["Position embeddings"] = position_embeddings[0][0].tolist()

    # Combined (what actually goes into the model)
    combined = token_embeddings + position_embeddings
    saving_dict["Combined embeddings"] = combined[0][0].tolist()

    first_layer_first_norm = model.h[0].ln_1(combined)
    saving_dict["First layer norm"] = first_layer_first_norm[0][0].tolist()

    with open("test/test_data.dump", "w") as f:
        json.dump(saving_dict, f)
