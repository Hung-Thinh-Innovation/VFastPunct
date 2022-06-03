from typing import List
from torch.utils.data import Dataset

import torch
import pandas as pd


class BaseFeatures(object):
    def __init__(self, input_ids, token_type_ids, attention_mask, valid_ids, label_masks):
        self.input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        self.token_type_ids = torch.as_tensor(token_type_ids, dtype=torch.long)
        self.attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        self.valid_ids = torch.as_tensor(valid_ids, dtype=torch.long)
        self.label_masks = torch.as_tensor(label_masks, dtype=torch.long)


class BaseCachedDataset(Dataset):
    def __init__(self, examples: List[BaseFeatures], device: str = 'cpu'):
        self.device = device
        self.examples = examples

    def __getitem__(self, index):
        return {key: val.to(self.device) for key, val in self.examples[index].__dict__.items()}

    def __len__(self):
        return len(self.examples)


class BaseDataset(Dataset):
    def __init__(self,
                 examples: pd.DataFrame,
                 tokenizer,
                 max_seq_len: int,
                 device: str = 'cpu',
                 use_crf: bool = False):
        self.device = device
        self.use_crf = use_crf
        self.max_len = max_seq_len
        self.tokenizer = tokenizer
        self.examples = examples

    def __getitem__(self, index):
        return NotImplementedError

    def __len__(self):
        return len(self.examples)