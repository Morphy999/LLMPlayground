import torch


def text_to_tokens_ids(text, tokenizer) -> torch.Tensor:
    encoded_text = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded_text).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(tokens_ids, tokenizer) -> str:
    flat = tokens_ids.squeeze(0)
    text = tokenizer.decode(flat.tolist())
    return text
