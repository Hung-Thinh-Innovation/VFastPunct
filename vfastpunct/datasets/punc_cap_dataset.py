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
        plabels = [PUNC_LABEL2ID.index(l) for l in self.examples.labels[index].split()]
        clabels = [CAP_LABEL2ID.index(l) for l in self.examples.labels[index].split()]
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
        label_masks.extend([0] * label_padding_size)

        encoding.pop('offset_mapping', None)
        item = {key: torch.as_tensor(val).to(self.device, dtype=torch.long) for key, val in encoding.items()}
        item['plabels'] = torch.as_tensor(plabels).to(self.device, dtype=torch.long)
        item['valid_ids'] = torch.as_tensor(valid_id).to(self.device, dtype=torch.long)
        item['label_masks'] = torch.as_tensor(label_masks).to(self.device, dtype=torch.long)
        return item

    def __len__(self):
        return len(self.examples)


def build_dataset(data_dir,
                  tokenizer,
                  data_type: str = 'train',
                  max_seq_length: int = 128,
                  overwrite_data: bool = False,
                  device: str = 'cpu'):
    data_file = Path(data_dir + f'/{data_type}.txt')
    data_splitted_file = Path(data_dir + f'/{data_type}_splitted.txt')
    assert os.path.exists(data_file) or os.path.exists(data_splitted_file), \
        f'`{data_file}` not exists! Do you realy have a dataset?'
    if not os.path.exists(data_splitted_file) or overwrite_data:
        LOGGER.info("Slipt dataset to example. It will take a few minutes, calm down and wait!")
        data_df = pd.read_csv(data_file, encoding='utf-8', sep=' ', names=['token', 'label'],
                              keep_default_na=False)
        data_df = split_example(data_df, EOS_MARKS, max_len=max_seq_length)
        data_df.to_csv(data_splitted_file)
    else:
        data_df = pd.read_csv(data_splitted_file)
    punc_dataset = PuncCapDataset(data_df, tokenizer, max_len=max_seq_length, device=device)
    return punc_dataset
