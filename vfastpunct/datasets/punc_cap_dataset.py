from torch.utils.data import Dataset
from pathlib import Path

from vfastpunct.constants import LOGGER, EOS_MARKS, PUNC_LABEL2ID, CAP_LABEL2ID
from vfastpunct.processor import normalize_text, split_example

import os
import torch
import pandas as pd
import numpy as np


class PuncCapDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 tokenizer,
                 max_len: int,
                 device: str = 'cpu'):
        self.examples = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    def __getitem__(self, index):
        sentence = self.examples.example[index]
        plabels = [PUNC_LABEL2ID.index(l) for l in self.examples.plabels[index].split()]
        clabels = [CAP_LABEL2ID.index(l) for l in self.examples.clabels[index].split()]
        label_masks = [1] * len(plabels)
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
        label_padding_size = (self.max_len - len(plabels))
        plabels.extend([0] * label_padding_size)
        clabels.extend([0] * label_padding_size)
        label_masks.extend([0] * label_padding_size)

        encoding.pop('offset_mapping', None)
        item = {key: torch.as_tensor(val).to(self.device, dtype=torch.long) for key, val in encoding.items()}
        item['plabels'] = torch.as_tensor(plabels).to(self.device, dtype=torch.long)
        item['clabels'] = torch.as_tensor(clabels).to(self.device, dtype=torch.long)
        item['valid_ids'] = torch.as_tensor(valid_id).to(self.device, dtype=torch.long)
        item['label_masks'] = torch.as_tensor(label_masks).to(self.device, dtype=torch.long)
        return item

    def __len__(self):
        return len(self.examples)


def build_punccap_dataset(dfile,
                  tokenizer,
                  data_type: str = 'train',
                  max_seq_length: int = 128,
                  overwrite_data: bool = False,
                  device: str = 'cpu'):
    data_df = pd.read_csv(dfile)
    punc_dataset = PuncCapDataset(data_df, tokenizer, max_len=max_seq_length, device=device)
    return punc_dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, RandomSampler
    d = build_punccap_dataset('/media/datngo/Data4/puncdataset/splitted/train_000_splitted.txt',
                          tokenizer=AutoTokenizer.from_pretrained('bert-base-multilingual-cased'),
                          max_seq_length=190,
                          device='cuda')

    train_sampler = RandomSampler(d)
    train_iterator = DataLoader(d, sampler=train_sampler, batch_size=32, num_workers=0)
    for b in train_iterator:
        print(b)

