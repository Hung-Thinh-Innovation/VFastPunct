from typing import List, Union
from pathlib import Path
from string import punctuation
from vfastpunct.constants import STRPUNC_MAPPING, PUNC_MAPPING, CAP_MAPPING
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


def get_total_lines(fpath: Union[str, os.PathLike]):
    return int(subprocess.check_output(["wc", "-l", fpath]).split()[0])


def train_test_split(fpath: Union[str, os.PathLike], total_lines: int, test_ratio=0.2):
    out_path = str(Path(fpath).parent) + '/raw_'
    cmd = f'shuf {fpath} | split -a1 -d  -l $({int(total_lines * (1 - test_ratio))}) - {out_path}'
    subprocess.run(cmd)
    return Path(out_path+'0'), Path(out_path+'1')


def binary_search(a, x):
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return -1


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
    result = re.sub(f'(\d) *([{punctuation}]) *(\d)', r'\1\2\3', result)
    return result.strip()


def make_dataset(data_file: Union[str, os.PathLike], split_test=False, test_ratio: float = 0.2, debug: bool = False):
    punct_pattern = re.compile(f'[{punctuation}]+')
    raw_path = Path(data_file)
    debug_count = 0
    total_lines = get_total_lines(raw_path)
    dataset_mapping = {'train.txt': (raw_path, total_lines), 'test.txt': None}

    if split_test:
        raw_train_path, raw_test_path = train_test_split(raw_path, total_lines=total_lines, test_ratio=test_ratio)
        num_train = int(total_lines * (1 - test_ratio))
        dataset_mapping = {
            'train.txt': (raw_train_path, num_train),
            'test.txt': (raw_test_path, total_lines - num_train)
        }
    for dtype, (dpath, num_line) in dataset_mapping.items():
        if dpath is None:
            continue
        dwriter = open(Path(str(raw_path.parent) + f'/{dtype}'), 'w')
        with open(dpath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(tqdm(f, total=num_line)):
                cur_examples = []
                line = normalize_text(line)
                matches = punct_pattern.finditer(line)
                end_idx = 0
                for m in matches:
                    tokens = line[end_idx: m.start()].split()
                    if len(tokens) == 0:
                        end_idx = m.end()
                        continue
                    for t in tokens[:-1]:
                        cur_examples.extend([(t.lower(), 'O', get_cap_label(t))])
                    puncs = line[m.start(): m.end()]
                    punc_label = 'O'
                    if puncs in STRPUNC_MAPPING:
                        punc_label = STRPUNC_MAPPING[puncs]
                    else:
                        for punc in list(puncs):
                            if punc in STRPUNC_MAPPING:
                                punc_label = STRPUNC_MAPPING[punc]
                                break
                    cur_examples.append((tokens[-1].lower(), punc_label, get_cap_label(tokens[-1])))
                    end_idx = m.end()
                if not restoration_punct(cur_examples) == line and debug:
                    debug_count += 1
                    print(f"===== Error Case {debug_count} =====")
                    print(line)
                    print(restoration_punct(cur_examples))
                for example in cur_examples:
                    dwriter.write(' '.join(example)+'\n')
        dwriter.close()


# DEBUG
if __name__ == "__main__":
    raw_data_path = './datasets/Raw/corpus-full.txt'
    make_dataset(raw_data_path, split_test=True, debug=False)
