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
