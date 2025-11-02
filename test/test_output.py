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
model.eval()
with torch.no_grad():
    saving_dict = dict()
    # Token embeddings
    token_embeddings = model.transformer.wte(input_ids)
    saving_dict["Token embeddings"] = token_embeddings[0][0].tolist()

    # Positional embeddings
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length).unsqueeze(0)
    position_embeddings = model.transformer.wpe(position_ids)
    saving_dict["Position embeddings"] = position_embeddings[0][0].tolist()

    # Combined (what actually goes into the model)
    input_embedding = token_embeddings + position_embeddings
    saving_dict["Combined embeddings"] = input_embedding[0][0].tolist()

    first_layer_first_norm = model.transformer.h[0].ln_1(input_embedding)  # pyright: ignore[reportCallIssue]
    first_layer_attention_exp = model.transformer.h[0].attn.c_attn(  # pyright: ignore[reportAttributeAccessIssue, reportCallIssue]
        first_layer_first_norm
    )  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
    saving_dict["First layer norm"] = first_layer_first_norm[0][0].tolist()
    saving_dict["First layer attention-exp"] = first_layer_attention_exp[0][0].tolist()
    first_layer_attention = model.transformer.h[0].attn(
        first_layer_first_norm,
    )  # pyright: ignore[reportCallIssue]
    saving_dict["First layer attention"] = first_layer_attention[0][0][0].tolist()
    first_layer_first_norm2 = model.transformer.h[0].ln_2(
        input_embedding + first_layer_attention[0]
    )  # pyright: ignore[reportCallIssue]
    saving_dict["First layer norm 2"] = first_layer_first_norm2[0][0].tolist()
    first_layer_mlp_1 = model.transformer.h[0].mlp.c_fc(first_layer_first_norm2)  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
    saving_dict["First layer mlp 1"] = first_layer_mlp_1[0][0].tolist()
    first_layer_gelu = torch.nn.GELU()(first_layer_mlp_1)
    saving_dict["First layer gelu"] = first_layer_gelu[0][0].tolist()
    print(torch.nn.GELU())
    first_layer_mlp_2 = model.transformer.h[0].mlp.c_proj(first_layer_gelu)  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
    saving_dict["First layer mlp 2"] = first_layer_mlp_2[0][0].tolist()
    first_layer_full_manual = (
        model.transformer.h[0].mlp(first_layer_first_norm2)  # pyright: ignore[reportCallIssue]
        + first_layer_attention[0]
        + input_embedding  # pyright: ignore[reportCallIssue]
    )

    saving_dict["First layer full manual"] = first_layer_full_manual[0][0].tolist()

    first_layer_full = model.transformer.h[0](input_embedding)  # pyright: ignore[reportCallIssue]
    saving_dict["First layer full"] = first_layer_full[0][0][0].tolist()

    last_full_output = first_layer_full[0]
    for i in range(1, 12):
        last_full_output = model.transformer.h[i](last_full_output)[0]

    hidden_states = model.transformer.ln_f(last_full_output)
    logits = model.lm_head(hidden_states)
    last_token_logits = logits[0, -1, :]  # [50257]
    next_token_id = torch.argmax(last_token_logits).item()
    next_token = tokenizer.decode([next_token_id])
    print("Last full output: ", last_full_output[0][0][0])
    print("After last ln : ", hidden_states[0])
    print("After last linear: ", logits[0][0])
    print(next_token_id)
    print(next_token)

    print(
        "MLP input: ",
        first_layer_first_norm2[0][0][0].item(),  # pyright: ignore[reportCallIssue]
        "MLP output: ",
        model.transformer.h[0].mlp(first_layer_first_norm2)[0][0][0].item(),  # pyright: ignore[reportCallIssue]
        "skip connection from attention: ",
        first_layer_attention[0][0][0][0].item(),
        "Result sum manual",
        first_layer_full_manual[0][0][0],
        "Result sum auto",
        first_layer_full[0][0][0][0],
    )

    with open("test/test_data.dump", "w") as f:
        json.dump(saving_dict, f)
