import random
from typing import List, Union
from pathlib import Path
from string import punctuation
from vfastpunct.constants import STRPUNC_MAPPING, PUNC_MAPPING, CAP_MAPPING, EOS_MARKS
from tqdm import tqdm

import re
import bisect
import os
import subprocess
import pandas as pd


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


def split_example_from_file(dpath: Union[str or os.PathLike], eos_marks: List[str], max_len: int = 128):
    idx = 0
    dataframe = pd.read_csv(dpath, encoding='utf-8', sep=' ', names=['token', 'plabel', 'clabel'], keep_default_na=False)
    num_token = len(dataframe)
    examples = []
    while num_token > idx >= 0:
        sub_data = dataframe[idx: min(idx + max_len, num_token)]
        end_idx = sub_data[sub_data.plabel.isin(eos_marks)].tail(1).index
        if end_idx.empty:
            end_idx = -1
            example_df = dataframe.iloc[idx:]
        else:
            end_idx = end_idx.item() + 1
            example_df = dataframe.iloc[idx: end_idx]
        examples.append([" ".join(example_df.token.values.tolist()),
                         " ".join(example_df.plabel.values.tolist()),
                         " ".join(example_df.clabel.values.tolist())])
        idx = end_idx
    return pd.DataFrame(examples, columns=["example", "plabels", "clabels"])


def get_total_lines(fpath: Union[str, os.PathLike]):
    return int(subprocess.check_output(["wc", "-l", fpath]).split()[0])


def truncate_file(fpath, sub_file_size: float = 2):
    p = Path(fpath)
    prefix = str(p.with_suffix('')) +'_'
    # subprocess.call(f'split -a 2 -b {sub_file_size}G -d {str(fpath)} {prefix}')
    subprocess.call(f'split -a 2 -b 2G -d ./datasets/Raw/train.txt ./datasets/Raw/train')


def binary_search(a, x):
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return -1


def get_cap_label(token: str) -> str:
    if token.isupper():
        return 'U'
    elif token.istitle():
        return 'T'
    else:
        return 'O'


def restoration_punct(examples: List):
    result = ''
    for t in examples:
        result += f'{CAP_MAPPING[t[-1]](t[0])}{PUNC_MAPPING[t[1]]} '
    result = re.sub(f'(\d) *([{punctuation}]) *(\d)', r'\1\2\3', result)
    return result.strip()


def make_dataset(data_file: Union[str, os.PathLike],
                 split_test=False,
                 test_ratio: float = 0.2,
                 is_truncate: bool = False,
                 truncate_size: int = 80000000):
    punct_pattern = re.compile(f'[{punctuation}]+')
    raw_path = Path(data_file)
    train_trunc_id, test_trunc_id = 0, 0
    train_count, test_count = 0, 0
    total_lines = get_total_lines(raw_path)
    train_writer = open(Path(str(raw_path.parent) + f'/train_{train_trunc_id:03}.txt'), 'w')
    test_writer = open(Path(str(raw_path.parent) + f'/test_{test_trunc_id:03}.txt'), 'w') if split_test else None
    with open(raw_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=total_lines)):
            cur_examples = ''
            cur_count = 0
            line = normalize_text(line)
            if len(line.split()) < 10:
                continue
            if len(punct_pattern.findall(line)) < 2 and random.uniform(0, 1) < 0.5:
                continue
            matches = punct_pattern.finditer(line)
            end_idx = 0
            for m in matches:
                tokens = line[end_idx: m.start()].split()
                if len(tokens) == 0:
                    end_idx = m.end()
                    continue
                for t in tokens[:-1]:
                    cur_examples += ' '.join([t.lower(), 'O', get_cap_label(t)]) + '\n'
                puncs = line[m.start(): m.end()]
                punc_label = 'O'
                if puncs in STRPUNC_MAPPING:
                    punc_label = STRPUNC_MAPPING[puncs]
                else:
                    for punc in list(puncs):
                        if punc in STRPUNC_MAPPING:
                            punc_label = STRPUNC_MAPPING[punc]
                            break
                cur_examples += ' '.join([tokens[-1].lower(), punc_label, get_cap_label(tokens[-1])]) + '\n'
                end_idx = m.end()
                cur_count += len(tokens)
            if random.uniform(0, 1) > test_ratio or not split_test:
                train_count += cur_count
                train_writer.write(cur_examples)
                if is_truncate and train_count >= truncate_size:
                    train_count = 0
                    train_trunc_id += 1
                    train_writer.close()
                    train_writer = open(Path(str(raw_path.parent) + f'/train_{train_trunc_id:03}.txt'), 'w')
            else:
                test_count += cur_count
                test_writer.write(cur_examples)
                test_count += 1
                if is_truncate and test_count >= truncate_size:
                    test_count = 0
                    test_trunc_id += 1
                    test_writer.close()
                    test_writer = open(Path(str(raw_path.parent) + f'/train_{test_trunc_id:03}.txt'), 'w')
    train_writer.close()
    test_writer.close()


def visualize_dataset(dpath: Union[str or os.PathLike]):
    import matplotlib.pyplot as plt
    import dask.dataframe as dd
    df = dd.read_csv(dpath, sep=' ', names=['word', 'plabel', 'clabel'])
    plt.figure(figsize=(10, 7))
    df[df['plabel'] != 'O'].plabel.value_counts().compute().plot(kind='bar')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('PLabel Distribution')
    plt.show()
    df.clabel.value_counts().compute().plot(kind='bar')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('CLabel Distribution')
    plt.show()


def split_examples(ddir: Union[str or os.PathLike]):
    import glob
    fpattern = str(Path(ddir + '/*_*.txt'))
    for f in tqdm(glob.glob(fpattern)):
        data_splitted_file = Path(f+'_splitted.txt')
        df = split_example_from_file(f, eos_marks=EOS_MARKS, max_len=190)
        df.to_csv(data_splitted_file)


# DEBUG
if __name__ == "__main__":
    make_dataset('/media/datngo/Data4/puncdataset/corpus-full.txt', split_test=True, is_truncate=True)
    split_examples('/media/datngo/Data4/puncdataset/')

