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


class SMSDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=None, padding_token_id=50256):
        self.df = df

        self.encoded_texts = [tokenizer.encode(text) for text in self.df["Text"]]

        self.labels = self.df["Label"].tolist()

        if max_length is None:
            self.max_length = self.get_max_lenght()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[: self.max_length] for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = self.pad_sequences(
            self.encoded_texts, self.max_length, padding_token_id
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

    def get_max_lenght(self):
        return max(len(encoded_text) for encoded_text in self.encoded_texts)

    def pad_sequences(self, sequences, max_length, padding_token_id):
        return [
            sequence + [padding_token_id] * (max_length - len(sequence)) for sequence in sequences
        ]
