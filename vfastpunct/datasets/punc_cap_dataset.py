from typing import Union
from pathlib import Path

from torch.utils.data import Dataset
from tqdm import tqdm
from vfastpunct.constants import LOGGER, EOS_MARKS, PUNC_LABEL2ID, CAP_LABEL2ID
from vfastpunct.processor import normalize_text, split_example

import os
import torch
import pandas as pd
import numpy as np


class PuncCapFeatures(object):
    def __init__(self, input_ids, token_type_ids, attention_mask, plabels, clabels, valid_ids, label_masks):
        self.input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        self.token_type_ids = torch.as_tensor(token_type_ids, dtype=torch.long)
        self.attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        self.attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        self.plabels = torch.as_tensor(plabels, dtype=torch.long)
        self.clabels = torch.as_tensor(clabels, dtype=torch.long)
        self.valid_ids = torch.as_tensor(valid_ids, dtype=torch.long)
        self.label_masks = torch.as_tensor(label_masks, dtype=torch.long)


class PuncCapDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 tokenizer,
                 max_len: int,
                 device: str = 'cpu'):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        self.examples = self.convert_example_to_feature(data)

    def convert_example_to_feature(self, data:pd.DataFrame):
        examples = []
        for index, row in tqdm(data.iterrows(), total=len(data)):
            sentence = row.example
            plabels = [PUNC_LABEL2ID.index(l) for l in row.plabels.split()]
            clabels = [CAP_LABEL2ID.index(l) for l in row.clabels.split()]
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
            items = {key: val for key, val in encoding.items()}
            items['plabels'] = plabels
            items['clabels'] = clabels
            items['valid_ids'] = valid_id
            items['label_masks'] = label_masks
            examples.append(PuncCapFeatures(**items))
        return examples

    def __getitem__(self, index):
        return {key: val.to(self.device) for key, val in self.examples[index].__dict__.items()}

    def __len__(self):
        return len(self.examples)


def build_punccap_dataset(dfile: Union[str, os.PathLike],
                  tokenizer,
                  data_type: str = 'train',
                  max_seq_length: int = 128,
                  overwrite_data: bool = False,
                  device: str = 'cpu'):
    dfile = Path(dfile)
    cached_features_file = dfile.with_suffix('.cached')
    if os.path.exists(cached_features_file):
        punc_dataset = torch.load(cached_features_file)
        punc_dataset.device=device
    else:
        LOGGER.info("Creating features from dataset file at %s", dfile)
        data_df = pd.read_csv(dfile)
        punc_dataset = PuncCapDataset(data_df, tokenizer, max_len=max_seq_length, device=device)
        LOGGER.info("Saving features into cached file %s", cached_features_file)
        torch.save(punc_dataset, cached_features_file)
    return punc_dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, RandomSampler
    d = build_punccap_dataset('./datasets/Raw/samples/train_000_splitted.txt',
                          tokenizer=AutoTokenizer.from_pretrained('bert-base-multilingual-cased'),
                          max_seq_length=190,
                          device='cpu')

    train_sampler = RandomSampler(d)
    train_iterator = DataLoader(d, sampler=train_sampler, batch_size=32, num_workers=0)
    for b in train_iterator:
        print(b)

