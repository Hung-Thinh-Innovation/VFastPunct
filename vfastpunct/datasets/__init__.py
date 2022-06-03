from vfastpunct.constants import LOGGER
from vfastpunct.datasets.base_dataset import BaseCachedDataset
from vfastpunct.datasets.punc_dataset import convert_punct_examples_to_feature, PunctDataset
from vfastpunct.datasets.punc_cap_dataset import convert_punctcap_examples_to_feature, PunctCapDataset

from typing import Union
from pathlib import Path

import os
import torch
import pandas as pd

CONVERT_FUNCT_MAPPING = {
    'punct': convert_punct_examples_to_feature,
    'punctcap': convert_punctcap_examples_to_feature
}

DATASET_MAPPING = {
    'punct':  PunctDataset,
    'punctcap':  PunctCapDataset
}


def build_and_cached_dataset(dfile_path: Union[str, os.PathLike],
                            tokenizer,
                            task: str = 'punct',
                            max_seq_length: int = 128,
                            use_crf: bool = False,
                            cached_dataset: bool = False,
                            overwrite_data: bool = False,
                            device: str = 'cpu'):
    dfile_path = Path(dfile_path)
    if cached_dataset:
        cached_path = dfile_path.with_suffix('.cached')
        if not os.path.exists(cached_path):
            LOGGER.info("Read examples from dataset file at %s", dfile_path)
            data_df = pd.read_csv(dfile_path)
            LOGGER.info("Creating features from dataset file at %s", dfile_path)
            feats = CONVERT_FUNCT_MAPPING[task](data_df,
                                                tokenizer,
                                                max_len=max_seq_length,
                                                use_crf=use_crf)
            LOGGER.info("Cached features from cached file at %s", cached_path)
            torch.save(feats, cached_path)
        else:
            LOGGER.info("Load features from cached file at %s", cached_path)
            feats = torch.load(cached_path)
        return BaseCachedDataset(feats, device=device)
    else:
        LOGGER.info("Read examples from dataset file at %s", dfile_path)
        data_df = pd.read_csv(dfile_path)
        dataset = DATASET_MAPPING[task](data_df,
                                        tokenizer=tokenizer,
                                        max_seq_len=max_seq_length,
                                        device=device,
                                        use_crf=use_crf)
        return dataset
