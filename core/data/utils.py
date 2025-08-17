import numpy as np
import torch


def text_to_token_ids(text, tokenizer) -> torch.Tensor:
    encoded_text = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded_text)
    return encoded_tensor


def token_ids_to_text(tokens_ids, tokenizer) -> str:
    flat = tokens_ids
    text = tokenizer.decode(flat.tolist())
    return text


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)  # 2 passo

        logits = logits[
            :, -1, :
        ]  # (batch, n_token, vocab_size) ->(batch_size, vocab_size)  # 3 passo

        probs = torch.softmax(logits, dim=-1)  # 4 passo, convertendo para probabilidades

        idx_next = torch.argmax(
            probs, dim=-1, keepdim=True
        )  # 5 passo, pegando o indice do token mais provavel

        idx = torch.cat(
            (idx, idx_next), dim=1
        )  # 6 passo, concatenando o indice do token mais provavel com o contexto

    return idx


def generate_text(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_k_probs, _ = torch.topk(logits, top_k)
            logits = torch.where(
                logits < top_k_probs[:, -1], torch.tensor(float("-inf")).to(logits.device), logits
            )

        if temperature > 0.0:
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, " "Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb_layer.weight = assign(gpt.pos_emb_layer.weight, params["wpe"])
    gpt.emb_layer.weight = assign(gpt.emb_layer.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformer_blocks[b].attn_layer.Wq.weight = assign(
            gpt.transformer_blocks[b].attn_layer.Wq.weight, q_w.T
        )
        gpt.transformer_blocks[b].attn_layer.Wk.weight = assign(
            gpt.transformer_blocks[b].attn_layer.Wk.weight, k_w.T
        )
        gpt.transformer_blocks[b].attn_layer.Wv.weight = assign(
            gpt.transformer_blocks[b].attn_layer.Wv.weight, v_w.T
        )
        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transformer_blocks[b].attn_layer.Wq.bias = assign(
            gpt.transformer_blocks[b].attn_layer.Wq.bias, q_b
        )
        gpt.transformer_blocks[b].attn_layer.Wk.bias = assign(
            gpt.transformer_blocks[b].attn_layer.Wk.bias, k_b
        )
        gpt.transformer_blocks[b].attn_layer.Wv.bias = assign(
            gpt.transformer_blocks[b].attn_layer.Wv.bias, v_b
        )
        gpt.transformer_blocks[b].attn_layer.out_proj.weight = assign(
            gpt.transformer_blocks[b].attn_layer.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )

        gpt.transformer_blocks[b].attn_layer.out_proj.bias = assign(
            gpt.transformer_blocks[b].attn_layer.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )
        gpt.transformer_blocks[b].ff_layer.fc1.weight = assign(
            gpt.transformer_blocks[b].ff_layer.fc1.weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.transformer_blocks[b].ff_layer.fc1.bias = assign(
            gpt.transformer_blocks[b].ff_layer.fc1.bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.transformer_blocks[b].ff_layer.fc2.weight = assign(
            gpt.transformer_blocks[b].ff_layer.fc2.weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.transformer_blocks[b].ff_layer.fc2.bias = assign(
            gpt.transformer_blocks[b].ff_layer.fc2.bias, params["blocks"][b]["mlp"]["c_proj"]["b"]
        )
        gpt.transformer_blocks[b].norm_layer1.gamma = assign(
            gpt.transformer_blocks[b].norm_layer1.gamma, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.transformer_blocks[b].norm_layer1.beta = assign(
            gpt.transformer_blocks[b].norm_layer1.beta, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.transformer_blocks[b].norm_layer2.gamma = assign(
            gpt.transformer_blocks[b].norm_layer2.gamma, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.transformer_blocks[b].norm_layer2.beta = assign(
            gpt.transformer_blocks[b].norm_layer2.beta, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.ln_f.gamma = assign(gpt.ln_f.gamma, params["g"])
    gpt.ln_f.beta = assign(gpt.ln_f.beta, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
