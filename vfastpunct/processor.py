from vfastpunct.constants import LOGGER, PUNCT_PATTERN, STRPUNC_MAPPING, PUNCT_MAPPING, CAP_MAPPING, EOS_MARKS
from vfastpunct.arguments import get_build_dataset_argument, get_split_argument

from typing import List, Union
from tqdm import tqdm
from pathlib import Path
from string import punctuation
from prettytable import PrettyTable

import re
import os
import sys
import glob
import bisect
import random
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
            end_idx = sub_data.index[-1] + 1
            example_df = sub_data
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
        result += f'{CAP_MAPPING[t[-1]](t[0])}{PUNCT_MAPPING[t[1]]} '
    result = re.sub(f'(\d) *([{punctuation}]) *(\d)', r'\1\2\3', result)
    return result.strip()


def build_dataset():
    args = get_build_dataset_argument()
    LOGGER.info(f"{'=' * 20}BUILD SUMMARY{'=' * 20}")
    summary_table = PrettyTable(["Arguments", "Values"])
    summary_table.add_rows([['Corpus', args.corpus_path],
                            ['Split test', args.split_test],
                            ['Test ratio', args.test_ratio],
                            ['Truncate', args.truncate],
                            ['Truncate size', args.truncate_size],
                            ['Skip ratio', args.skip_ratio]])
    LOGGER.info(summary_table)
    raw_path = Path(args.corpus_path)
    train_trunc_id, test_trunc_id = 0, 0
    train_count, test_count = 0, 0
    total_lines = get_total_lines(raw_path)
    train_writer = open(Path(str(raw_path.parent) + f'/train_{train_trunc_id:03}.txt'), 'w')
    test_writer = open(Path(str(raw_path.parent) + f'/test_{test_trunc_id:03}.txt'), 'w') if args.split_test else None
    with open(raw_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=total_lines)):
            if random.uniform(0, 1) < args.skip_ratio:
                continue
            cur_examples = ''
            cur_count = 0
            line = normalize_text(line)
            if len(line.split()) < 10:
                continue
            if len(PUNCT_PATTERN.findall(line)) < 2 and random.uniform(0, 1) < 0.5:
                continue
            matches = PUNCT_PATTERN.finditer(line)
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
            if random.uniform(0, 1) > args.test_ratio or not args.split_test:
                train_count += cur_count
                train_writer.write(cur_examples)
                if args.truncate and train_count >= args.truncate_size:
                    train_count = 0
                    train_trunc_id += 1
                    train_writer.close()
                    train_writer = open(Path(str(raw_path.parent) + f'/train_{train_trunc_id:03}.txt'), 'w')
            else:
                test_count += cur_count
                test_writer.write(cur_examples)
                test_count += 1
                if args.truncate and test_count >= args.truncate_size:
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


def split_examples():
    args = get_split_argument()
    LOGGER.info(f"{'=' * 20}SPLIT SUMMARY{'=' * 20}")
    summary_table = PrettyTable(["Arguments", "Values"])
    summary_table.add_rows([['Data dir', args.data_dir],
                            ['Max lenght', args.max_len]])
    LOGGER.info(summary_table)
    fpattern = str(Path(args.data_dir + '/*.txt'))
    dfiles = glob.glob(fpattern)
    if len(dfiles) == 0:
        LOGGER.info(f"No found any dataset file in {args.data_dir}; Notice: Only read `.txt` file.")
        return None
    tqdm_bar = tqdm(dfiles, desc="Spliting")
    for idx, f in enumerate(tqdm_bar):
        data_splitted_file = re.sub('\.txt', '_splitted.txt', f)
        df = split_example_from_file(f, eos_marks=EOS_MARKS, max_len=args.max_len)
        df.to_csv(data_splitted_file)
    LOGGER.info("Hey, DONE !!!!!!")


# DEBUG
if __name__ == "__main__":
    if sys.argv[1] == 'build':
        LOGGER.info("Start BUILD dataset process... go go go!!!")
        build_dataset()
    elif sys.argv[1] == 'split':
        LOGGER.info("Start SPLIT dataset process... go go go!!!")
        split_examples()
    else:
        LOGGER.error(
            f'[ERROR] - `{sys.argv[1]}` Are you kidding me? I only know `build` or `split`. Please read the README!!!!')
