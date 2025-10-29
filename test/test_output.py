import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Input text
text = "The main character of The lord of the rings is "
input_ids = tokenizer.encode(text, return_tensors="pt")
print(input_ids[0][0])

# Get embeddings (output of embedding layer)
with torch.no_grad():
    saving_dict = dict()
    # Token embeddings
    token_embeddings = model.transformer.wte(input_ids)
    # print("Token embeddings shape:", token_embeddings.shape)
    print("Token embeddings:\n", token_embeddings[0][0])
    saving_dict["Token embeddings"] = token_embeddings[0][0].tolist()

    # Positional embeddings
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length).unsqueeze(0)
    position_embeddings = model.transformer.wpe(position_ids)
    # print("\nPosition embeddings shape:", position_embeddings.shape)
    print("Position embeddings:\n", position_embeddings[0][0])
    saving_dict["Position embeddings"] = position_embeddings[0][0].tolist()

    # Combined (what actually goes into the model)
    combined = token_embeddings + position_embeddings
    print("\nCombined embeddings:\n", combined[0][0])
    saving_dict["Combined embeddings"] = combined[0][0].tolist()
    with open("test/test_data.dump", "w") as f:
        json.dump(saving_dict, f)
