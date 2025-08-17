import torch
from torch.utils.data import Dataset

from .utils import text_to_token_ids


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.text = text
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        self.input_ids = []
        self.target_ids = []

        self.tokens_ids = text_to_token_ids(self.text, self.tokenizer)

        for i in range(0, len(self.tokens_ids) - self.max_length, self.stride):
            input_ids = self.tokens_ids[i : i + self.max_length]
            target_ids = self.tokens_ids[i + 1 : i + self.max_length + 1]
            self.input_ids.append(input_ids)
            self.target_ids.append(target_ids)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
