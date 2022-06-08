from typing import Union
from pathlib import Path
from tqdm import tqdm

from vfastpunct.constants import LOGGER, PUNCT_LABEL2ID, CAP_LABEL2ID
from vfastpunct.processor import normalize_text
from vfastpunct.datasets.base_dataset import BaseFeatures, BaseDataset

import os
import torch
import pandas as pd
import numpy as np


class PunctCapFeatures(BaseFeatures):
    def __init__(self, plabels, clabels, **kwargs):
        super(PunctCapFeatures, self).__init__(**kwargs)
        self.plabels = torch.as_tensor(plabels, dtype=torch.long)
        self.clabels = torch.as_tensor(clabels, dtype=torch.long)


def convert_punctcap_examples_to_feature(data:pd.DataFrame, tokenizer, max_len: int = 190, use_crf: bool = False):
    examples = []
    for index, row in tqdm(data.iterrows(), total=len(data)):
        sentence = row.example
        plabels = [PUNCT_LABEL2ID.index(l) for l in row.plabels.split()]
        clabels = [CAP_LABEL2ID.index(l) for l in row.clabels.split()]
        label_masks = [1] * len(plabels)
        encoding = tokenizer(normalize_text(sentence),
                             padding='max_length',
                             truncation=True,
                             return_offsets_mapping=True,
                             max_length=max_len)

        valid_id = np.ones(len(encoding["offset_mapping"]), dtype=int)
        valid_plabels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        valid_clabels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping == (0, 0) or (mapping[0] != 0 and mapping[0] == encoding["offset_mapping"][idx - 1][-1]):
                valid_id[idx] = 0
                continue
            valid_plabels[idx] = plabels[i]
            valid_clabels[idx] = clabels[i]
            i += 1
        label_padding_size = (max_len - len(plabels))
        plabels.extend([0] * label_padding_size)
        clabels.extend([0] * label_padding_size)
        label_masks.extend([0] * label_padding_size)

        encoding.pop('offset_mapping', None)
        items = {key: val for key, val in encoding.items()}
        items['plabels'] = plabels if use_crf else valid_plabels
        items['clabels'] = clabels if use_crf else valid_clabels
        items['valid_ids'] = valid_id
        items['label_masks'] = label_masks if use_crf else valid_id
        examples.append(PunctCapFeatures(**items))
    return examples


class PunctCapDataset(BaseDataset):
    def __getitem__(self, index):
        example = self.examples.iloc[index]
        sentence = example.example
        plabels = [PUNCT_LABEL2ID.index(l) for l in example.plabels.split()]
        clabels = [CAP_LABEL2ID.index(l) for l in example.clabels.split()]
        if len(clabels) == 0:
            clabels = [0] * len(plabels)
        ex_len = len(plabels)
        label_masks = [1] * ex_len
        encoding = self.tokenizer(normalize_text(sentence),
                                  padding='max_length',
                                  truncation=True,
                                  return_offsets_mapping=True,
                                  max_length=self.max_len)

        valid_id = np.ones(len(encoding["offset_mapping"]), dtype=int)
        valid_plabels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        valid_clabels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping == (0, 0) or (mapping[0] != 0 and mapping[0] == encoding["offset_mapping"][idx - 1][-1]):
                valid_id[idx] = 0
                continue
            valid_plabels[idx] = plabels[i]
            valid_clabels[idx] = clabels[i]
            i += 1

        label_padding_size = (self.max_len - ex_len)
        plabels.extend([0] * label_padding_size)
        clabels.extend([0] * label_padding_size)
        label_masks.extend([0] * label_padding_size)

        encoding.pop('offset_mapping', None)
        items = {key: val for key, val in encoding.items()}
        items['plabels'] = plabels if self.use_crf else valid_plabels
        items['clabels'] = clabels if self.use_crf else valid_clabels
        items['valid_ids'] = valid_id
        items['label_masks'] = label_masks if self.use_crf else valid_id
        features = PunctCapFeatures(**items)
        return{key: val.to(self.device) for key, val in features.__dict__.items()}


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    data_df = pd.read_csv('./datasets/Raw/train_015_splitted.txt')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    d = PunctCapDataset(data_df,
                        device='cpu',
                        tokenizer=tokenizer,
                        max_seq_len=190,
                        use_crf=False)

    valid_iterator = DataLoader(d, batch_size=32, shuffle=True, num_workers=20)
    for b in tqdm(valid_iterator, total = len(valid_iterator)):
        continue