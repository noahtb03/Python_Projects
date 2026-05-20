"""
datasets.py
-----------
PyTorch Dataset classes and collators for both models.
"""

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding


class IncludeDataset(Dataset):
    """
    Dataset for the binary include model.
    Wraps texts and labels so PyTorch's DataLoader can iterate
    through them in batches. Each item gets tokenized on demand.
    """

    def __init__(self, texts, labels, tokenizer, max_len):
        # Store everything the dataset needs to produce one item
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        # PyTorch calls this to know the dataset size
        return len(self.texts)

    def __getitem__(self, idx):
        # PyTorch calls this to get one item by index
        # Converts text to token IDs; padding=False because the collator handles it per batch
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        # Attach the label to the encoding dictionary
        enc["labels"] = int(self.labels[idx])
        return enc


class MultiTaskDataset(Dataset):
    """
    Dataset for the multi-task classifier.
    Same as IncludeDataset but attaches 9 labels per item instead of 1,
    using prefixed keys like "label_ConcernRaised", "label_ContactReason".
    """

    def __init__(self, texts, labels_dict, tokenizer, max_len, fields):
        self.texts = texts
        self.labels_dict = labels_dict  # dict: {field_name: [list of int labels]}
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fields = fields  # list of all field names

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        # Attach all 9 labels with prefixed keys so the collator can separate them later
        for field in self.fields:
            enc[f"label_{field}"] = int(self.labels_dict[field][idx])
        return enc


class MultiTaskCollator:
    """
    Custom batch collator for the multi-task model.
    The standard Hugging Face collator pads token sequences but doesn't
    know about our 9 custom label fields. This separates labels from tokens,
    lets the standard collator pad the tokens, then reattaches labels as tensors.
    """

    def __init__(self, tokenizer, fields):
        self.pad_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self.fields = fields

    def __call__(self, features):
        # Separate label fields from token fields
        label_values = {f: [] for f in self.fields}
        token_features = []

        for feat in features:
            feat_copy = dict(feat)
            for f in self.fields:
                key = f"label_{f}"
                # .pop() removes the key and returns its value
                label_values[f].append(feat_copy.pop(key))
            token_features.append(feat_copy)

        # Let the standard collator pad the token sequences
        batch = self.pad_collator(token_features)

        # Reattach labels as tensors (dtype=long is what CrossEntropyLoss expects)
        for f in self.fields:
            batch[f"label_{f}"] = torch.tensor(label_values[f], dtype=torch.long)

        return batch


class PredictionDataset(Dataset):
    """
    Dataset for inference — no labels needed.
    Unlike training datasets, this pads every sequence to max_length
    upfront and returns PyTorch tensors directly since there's no
    collator involved during prediction.
    """

    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        # .squeeze(0) removes the extra batch dimension the tokenizer adds
        return {k: v.squeeze(0) for k, v in encoding.items()}