from vfastpunct.constants import PUNCT_LABEL2ID
from vfastpunct.processor import normalize_text
from vfastpunct.datasets.base_dataset import BaseFeatures, BaseDataset

from tqdm import tqdm

import torch
import numpy as np
import pandas as pd


class PunctFeatures(BaseFeatures):
    def __init__(self, plabels, **kwargs):
        super(PunctFeatures, self).__init__(**kwargs)
        self.plabels = torch.as_tensor(plabels, dtype=torch.long)


def convert_punct_examples_to_feature(data:pd.DataFrame, tokenizer, max_len: int = 190, use_crf: bool = False):
    examples = []
    for index, row in tqdm(data.iterrows(), total=len(data)):
        sentence = row.example
        plabels = [PUNCT_LABEL2ID.index(l) for l in row.plabels.split()]
        label_masks = [1] * len(plabels)
        encoding = tokenizer(normalize_text(sentence),
                             padding='max_length',
                             truncation=True,
                             return_offsets_mapping=True,
                             max_length=max_len)

        valid_id = np.ones(len(encoding["offset_mapping"]), dtype=int)
        valid_plabels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping == (0, 0) or (mapping[0] != 0 and mapping[0] == encoding["offset_mapping"][idx - 1][-1]):
                valid_id[idx] = 0
                continue
            valid_plabels[idx] = plabels[i]
            i += 1
        label_padding_size = (max_len - len(plabels))
        plabels.extend([0] * label_padding_size)
        label_masks.extend([0] * label_padding_size)

        encoding.pop('offset_mapping', None)
        items = {key: val for key, val in encoding.items()}
        items['plabels'] = plabels if use_crf else valid_plabels
        items['valid_ids'] = valid_id
        items['label_masks'] = label_masks if use_crf else valid_id
        examples.append(PunctFeatures(**items))
    return examples


class PunctDataset(BaseDataset):
    def __getitem__(self, index):
        example = self.examples.iloc[index]
        sentence = example.example
        plabels = [PUNCT_LABEL2ID.index(l) for l in example.plabels.split()]
        ex_len = len(plabels)
        label_masks = [1] * ex_len
        encoding = self.tokenizer(normalize_text(sentence),
                                  padding='max_length',
                                  truncation=True,
                                  return_offsets_mapping=True,
                                  max_length=self.max_len)

        valid_id = np.ones(len(encoding["offset_mapping"]), dtype=int)
        valid_plabels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping == (0, 0) or (mapping[0] != 0 and mapping[0] == encoding["offset_mapping"][idx - 1][-1]):
                valid_id[idx] = 0
                continue
            valid_plabels[idx] = plabels[i]
            i += 1

        label_padding_size = (self.max_len - ex_len)
        plabels.extend([0] * label_padding_size)
        label_masks.extend([0] * label_padding_size)

        encoding.pop('offset_mapping', None)
        items = {key: val for key, val in encoding.items()}
        items['plabels'] = plabels if self.use_crf else valid_plabels
        items['valid_ids'] = valid_id
        items['label_masks'] = label_masks if self.use_crf else valid_id
        features = PunctFeatures(**items)
        return{key: val.to(self.device) for key, val in features.__dict__.items()}