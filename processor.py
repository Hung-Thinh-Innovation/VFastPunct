from typing import List
from torch.utils.data import Dataset

import re
import torch
import pandas as pd
import numpy as np


def normalize_text(txt):
    txt = re.sub("\xad|\u200b", "", txt)
    return txt


class PuncDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 tokenizer,
                 max_len: int,
                 labels: List[str]):
        self.examples = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels

    def __getitem__(self, index):
        sentence = self.examples.example[index]
        labels = [self.labels.index(label) for label in self.examples.labels[index].split()]
        encoding = self.tokenizer(normalize_text(sentence),
                                  padding='max_length',
                                  truncation=True,
                                  return_offsets_mapping=True,
                                  max_length=self.max_len)
        valid_id = np.zeros(len(encoding["offset_mapping"]), dtype=int)
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping == (0, 0) or (mapping[0] != 0 and mapping[0] == encoding["offset_mapping"][idx - 1][-1]):
                continue
            valid_id[idx] = 1
            i += 1
        labels.extend([0] * (self.max_len - len(labels)))
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(labels)
        item['valid_ids'] = torch.as_tensor(valid_id)
        return item

    def __len__(self):
        return len(self.examples)
