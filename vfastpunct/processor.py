from typing import List, Union
from pathlib import Path
from torch.utils.data import Dataset
from string import punctuation
from sklearn.model_selection import train_test_split
from vfastpunct.constants import LOGGER, PUNC_LABEL2ID, EOS_MARKS, STRPUNC_MAPPING, PUNC_MAPPING, CAP_MAPPING

import re
import random
import os
import torch
import pandas as pd
import numpy as np


def normalize_text(txt):
    txt = re.sub("\xad|\u200b", "", txt)
    return txt.strip()


def split_example(dataframe: pd.DataFrame, eos_marks: List[str], max_len: int = 128):
    idx = 0
    num_token = len(dataframe)
    examples = []
    while num_token > idx >= 0:
        sub_data = dataframe[idx: min(idx + max_len, num_token)]
        end_idx = sub_data[sub_data.label.isin(eos_marks)].tail(1).index
        if end_idx.empty:
            end_idx = -1
            example_df = dataframe.iloc[idx:]
        else:
            end_idx = end_idx.item() + 1
            example_df = dataframe.iloc[idx: end_idx]
        examples.append([" ".join(example_df.token.values.tolist()), " ".join(example_df.label.values.tolist())])
        idx = end_idx
    return pd.DataFrame(examples, columns=["example", "labels"])

def get_cap_label(token: str) -> str:
    if token.isupper():
        return 'UPPER'
    elif token.istitle():
        return 'TITLE'
    else:
        return 'KEEP'

def restoration_punct(examples: List):
    result = ''
    for t in examples:
        result += f'{CAP_MAPPING[t[-1]](t[0])}{PUNC_MAPPING[t[1]]} '
    return result.strip()

def make_dataset(data_file: Union[str, os.PathLike], split_test=False, debug: bool = False):
    punct_pattern = re.compile(f'[{punctuation}]+')
    examples = []
    raw_path = Path(data_file)
    with open(raw_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            cur_example = []
            line = normalize_text(line)
            matches = punct_pattern.finditer(line)
            end_idx = 0
            for m in matches:
                tokens = line[end_idx: m.start()].split()
                if len(tokens) == 0:
                    end_idx = m.end()
                    continue
                for t in tokens[:-1]:
                    cur_example.extend([(t.lower(), 'O', get_cap_label(t))])
                puncs = line[m.start(): m.end()]
                punc_label = 'O'
                if puncs in STRPUNC_MAPPING:
                    punc_label = STRPUNC_MAPPING[puncs]
                else:
                    for punc in list(puncs):
                        if punc in STRPUNC_MAPPING:
                            punc_label = STRPUNC_MAPPING[punc]
                            break
                cur_example.append((tokens[-1].lower(), punc_label, get_cap_label(tokens[-1])))
                end_idx = m.end()
            examples.extend(cur_example)
            if not restoration_punct(cur_example) == line and debug:
                print(line)
                print(restoration_punct(cur_example))
                print('\n')
    df = pd.DataFrame(examples)
    if split_test:
        train_df, test_df = train_test_split(df, shuffle=False, test_size=0.2)
        train_df.to_csv(Path(str(raw_path.parent)+'/train.txt'), sep=' ', index=False, header=False)
        test_df.to_csv(Path(str(raw_path.parent)+'/test.txt'), sep=' ', index=False, header=False)
    else:
        df.to_csv(Path(str(raw_path.parent)+'/train.txt'), sep=' ', index=False, header=False)


class PuncDataset(Dataset):
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
        labels = [PUNC_LABEL2ID.index(l) for l in self.examples.labels[index].split()]
        label_masks = [1] * len(labels)
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
        label_padding_size = (self.max_len - len(labels))
        labels.extend([0] * label_padding_size)
        label_masks.extend([0] * label_padding_size)

        encoding.pop('offset_mapping', None)
        item = {key: torch.as_tensor(val).to(self.device, dtype=torch.long) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(labels).to(self.device, dtype=torch.long)
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
    punc_dataset = PuncDataset(data_df, tokenizer, max_len=max_seq_length, device=device)
    return punc_dataset


#DEBUG
if __name__ == "__main__":
    raw_data_path = './datasets/Raw/sample.txt'
    make_dataset(raw_data_path, split_test=True)
