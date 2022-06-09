from vfastpunct.constants import (LOGGER, PUNCT_PATTERN, CAP_PATTERN,
                                  STRPUNC_MAPPING, PUNCT_MAPPING, CAP_MAPPING, EOS_MARKS)
from vfastpunct.arguments import get_build_dataset_argument, get_split_argument

from typing import List, Union
from tqdm import tqdm
from pathlib import Path
from string import punctuation
from prettytable import PrettyTable
from tqdm.contrib.concurrent import thread_map

import re
import os
import sys
import glob
import time
import json
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
    dataframe = pd.read_csv(dpath, encoding='utf-8',
                            sep=' ',
                            names=['token', 'plabel', 'clabel'],
                            keep_default_na=False)
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
    elif CAP_PATTERN(token):
        return 'S'
    else:
        return 'O'


def restoration_punct(examples: List):
    result = ''
    for t in examples:
        result += f'{CAP_MAPPING[t[-1]](t[0])}{PUNCT_MAPPING[t[1]]} '
    result = re.sub(f'(\d) *([{punctuation}]) *(\d)', r'\1\2\3', result)
    return result.strip()


def _convert_plaint_text_to_example(line):
    examples = ''
    ptags, ctags = [], []
    num_token = 0
    case_dict = {}
    line = normalize_text(line)
    if len(line.split()) < 10:
        return None
    matches = PUNCT_PATTERN.finditer(line)
    end_idx = 0
    for m in matches:
        tokens = line[end_idx: m.start()].split()
        if len(tokens) == 0:
            end_idx = m.end()
            continue
        for t in tokens[:-1]:
            c_tag = get_cap_label(t)
            if c_tag == 'S':
                c_tag = 'T'
                case_dict[t.lower()] = t
            ptags.append('O')
            ctags.append(c_tag)
            examples += ' '.join([t.lower(), 'O', c_tag]) + '\n'
        puncts = line[m.start(): m.end()]
        punc_label = 'O'
        if puncts in STRPUNC_MAPPING:
            punc_label = STRPUNC_MAPPING[puncts]
        elif '...' in puncts and '...' in STRPUNC_MAPPING:
            punc_label = STRPUNC_MAPPING['...']
        else:
            for punc in list(puncts):
                if punc in STRPUNC_MAPPING:
                    punc_label = STRPUNC_MAPPING[punc]
                    break
        c_tag = get_cap_label(tokens[-1])
        if c_tag == 'S':
            c_tag = 'T'
            case_dict[tokens[-1].lower()] = tokens[-1]
        ptags.append(punc_label)
        ctags.append(c_tag)
        examples += ' '.join([tokens[-1].lower(), punc_label, c_tag]) + '\n'
        end_idx = m.end()
        num_token += len(tokens)
    return examples, ptags, ctags, num_token, case_dict


def _single_thread_build(raw_path, workers=0):
    total_lines = get_total_lines(raw_path)
    with open(raw_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=total_lines)):
            yield _convert_plaint_text_to_example(line)


def _multi_thread_build(raw_path, workers=2):
    with open(raw_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for outputs in thread_map(_convert_plaint_text_to_example, lines ,
                               max_workers=workers,
                               desc="Build dataset",
                               leave=False,
                               colour='blue'):
            yield outputs


def build_dataset_from_plain_text():
    args = get_build_dataset_argument()
    LOGGER.info(f"{'=' * 20}BUILD SUMMARY{'=' * 20}")
    summary_dict = {
        "config": vars(args),
        "stats": {
            "num_words": 0,
            "total_ptags": 0,
            "num_ptags": {tag: 0 for tag in PUNCT_MAPPING.keys()},
            "total_ctags": 0,
            "num_ctags":  {tag: 0 for tag in CAP_MAPPING.keys()}
        }
    }
    transform_case_dict = {}
    summary_table = PrettyTable(["Arguments", "Values"])
    summary_table.add_rows([[k, v] for k, v in summary_dict['config'].items()])
    LOGGER.info(summary_table)
    raw_path = Path(args.corpus_path)
    train_trunc_id, test_trunc_id = 0, 0
    train_count, test_count = 0, 0
    train_writer = open(Path(str(raw_path.parent) + f'/train_{train_trunc_id:03}.txt'), 'w')
    test_writer = open(Path(str(raw_path.parent) + f'/test_{test_trunc_id:03}.txt'), 'w') if args.split_test else None
    process_func =  _multi_thread_build if args.num_worker > 0 else _single_thread_build
    start_time = time.time()
    for examples in process_func(raw_path, workers=args.num_worker):
        if examples is None:
            continue
        cur_samples, cur_ptags, cur_ctags, num_token, case_dict = examples
        for k, v in case_dict.items():
            transform_case_dict[k] = v
        # Skip dataset with ratio
        if random.uniform(0, 1) < args.skip_ratio:
            rare_ptags = sorted(summary_dict['stats']['num_ptags'],
                                key=summary_dict['stats']['num_ptags'].get,
                                reverse=False)[:3]
            if len(set(cur_ptags) & set(rare_ptags)) == 0:
                continue

        summary_dict['stats']['num_words'] += num_token
        summary_dict['stats']['total_ptags'] += (num_token - cur_ptags.count("O"))
        for p_tag in cur_ptags: summary_dict['stats']['num_ptags'][p_tag] += 1
        summary_dict['stats']['total_ctags'] += (num_token - cur_ctags.count("O"))
        for c_tag in cur_ctags: summary_dict['stats']['num_ctags'][c_tag] += 1

        # Train-Test split
        if random.uniform(0, 1) > args.test_ratio or not args.split_test:
            train_count += num_token
            train_writer.write(cur_samples)
            if args.truncate and train_count >= args.truncate_size:
                train_count = 0
                train_trunc_id += 1
                train_writer.close()
                train_writer = open(Path(str(raw_path.parent) + f'/train_{train_trunc_id:03}.txt'), 'w')
        else:
            test_count += num_token
            test_writer.write(cur_samples)
            test_count += 1
            if args.truncate and test_count >= args.truncate_size:
                test_count = 0
                test_trunc_id += 1
                test_writer.close()
                test_writer = open(Path(str(raw_path.parent) + f'/train_{test_trunc_id:03}.txt'), 'w')
    spend_time = time.time() - start_time
    with open(os.path.join(raw_path.parent, "dataset_summary.json"), "w") as outfile:
        LOGGER.info(f"Save summary to `{raw_path.parent}/dataset_summary.json`")
        json.dump(summary_dict, outfile)
    with open(os.path.join(raw_path.parent, "special_case_dict.json"), "w") as outfile:
        LOGGER.info(f"Save transform case dict to `{raw_path.parent}/special_case_dict.json`")
        json.dump(transform_case_dict, outfile)
    LOGGER.info(f"Build process cost: {spend_time}")
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
        build_dataset_from_plain_text()
    elif sys.argv[1] == 'split':
        LOGGER.info("Start SPLIT dataset process... go go go!!!")
        split_examples()
    else:
        LOGGER.error(
            f'[ERROR] - `{sys.argv[1]}` Are you kidding me? I only know `build` or `split`. Please read the README!!!!')
