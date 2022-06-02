from typing import Union, List
from pathlib import Path

from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
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


def convert_example_to_feature(example, tokenizer, max_len: int = 190, use_crf: bool = False):
    sentence = example.example
    plabels = [PUNC_LABEL2ID.index(l) for l in example.plabels.split()]
    clabels = [CAP_LABEL2ID.index(l) for l in example.clabels.split()]
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
    return PuncCapFeatures(**items)


def multi_thread_convert_examples_to_feature(data:pd.DataFrame, tokenizer, max_len: int = 190, use_crf: bool = False):
    examples = []
    for index, row in tqdm(data.iterrows(), total=len(data)):
        sentence = row.example
        plabels = [PUNC_LABEL2ID.index(l) for l in row.plabels.split()]
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
        examples.append(PuncCapFeatures(**items))
    return examples


def convert_examples_to_feature(data:pd.DataFrame, tokenizer, max_len: int = 190, use_crf: bool = False):
    examples = []
    for index, row in tqdm(data.iterrows(), total=len(data)):
        sentence = row.example
        plabels = [PUNC_LABEL2ID.index(l) for l in row.plabels.split()]
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
        examples.append(PuncCapFeatures(**items))
    return examples


class PuncCapDataset(Dataset):
    def __init__(self, examples: List[PuncCapFeatures], device: str = 'cpu'):
        self.device = device
        self.examples = examples

    def __getitem__(self, index):
        return {key: val.to(self.device) for key, val in self.examples[index].__dict__.items()}

    def __len__(self):
        return len(self.examples)


class PunctCapDataset(Dataset):
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
        example = self.examples.iloc[index]
        sentence = example.example
        plabels = [PUNC_LABEL2ID.index(l) for l in example.plabels.split()]
        clabels = [CAP_LABEL2ID.index(l) for l in example.clabels.split()]
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
        features = PuncCapFeatures(**items)
        return{key: val.to(self.device) for key, val in features.__dict__.items()}

    def __len__(self):
        return len(self.examples)


def build_punccap_dataset(dfile: Union[str, os.PathLike],
                          tokenizer,
                          data_type: str = 'train',
                          max_seq_length: int = 128,
                          overwrite_data: bool = False,
                          device: str = 'cpu',
                          use_crf: bool = False):
    dfile = Path(dfile)
    LOGGER.info("Creating features from dataset file at %s", dfile)
    data_df = pd.read_csv(dfile)
    punccap_examples = convert_examples_to_feature(data_df, tokenizer, max_len=max_seq_length, use_crf=use_crf)
    return PuncCapDataset(punccap_examples, device=device)


def build_punctcap_dataset(dfile: Union[str, os.PathLike],
                          tokenizer,
                          data_type: str = 'train',
                          max_seq_length: int = 128,
                          overwrite_data: bool = False,
                          device: str = 'cpu',
                          use_crf: bool = False):
    dfile = Path(dfile)
    data_df = pd.read_csv(dfile)
    punctcap_dataset = PunctCapDataset(data_df,
                                       device=device,
                                       tokenizer=tokenizer,
                                       max_seq_len=max_seq_length,
                                       use_crf=use_crf)
    return punctcap_dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, RandomSampler

    data_df = pd.read_csv('./datasets/Raw/train_015_splitted.txt')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    # features = convert_example_to_feature(data_df, tokenizer, max_len=190, use_crf=False)
    # torch.save()
    # d = PuncCapDataset(features, device='cpu')
    d = PunctCapDataset(data_df,
                        device='cpu',
                        tokenizer=tokenizer,
                        max_seq_len=190,
                        use_crf=False)

    valid_iterator = DataLoader(d, batch_size=32, shuffle=True, num_workers=20)
    for b in tqdm(valid_iterator, total = len(valid_iterator)):
        continue

