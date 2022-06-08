from vfastpunct.constants import LOGGER, DATA_SOURCES, PUNCT_LABEL2ID, PUBLIC_PUNCT_LABEL2ID

from typing import List

import os
import requests
import random
import torch
import numpy as np


def download_file_from_google_drive(id, destination, confirm=None):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    URL = "https://docs.google.com/uc?export=download"
    if confirm is not None:
        URL += f"&confirm={confirm}"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def set_ramdom_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def map_labels(pred_labels: List[int], model_label2id: List[str], dataset_label2id: List[str]):
    if model_label2id == dataset_label2id:
        return pred_labels
    mapped_labels = []
    for pred_label in pred_labels:
        pred_tag = model_label2id[pred_label]
        pred_tag = pred_tag if pred_tag in dataset_label2id else 'O'
        mapped_labels.append(dataset_label2id.index(pred_tag))
    return mapped_labels


def get_total_model_parameters(model):
    total_params, trainable_params = 0, 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        if parameter.requires_grad:
            trainable_params += params
        total_params += params
    return total_params, trainable_params


def download_dataset_from_drive(save_dir):
    for idx, (k, v) in enumerate(DATA_SOURCES.items()):
        LOGGER.info(f"[{idx}/{len(DATA_SOURCES)}]Download {k} ...")
        save_path = os.path.join(save_dir, k)
        download_file_from_google_drive(v, save_path, 't')


if __name__ == "__main__":
    pred_labels = [0, 1, 2, 3, 4, 5, 6,7, 8, 8, 0]
    print(map_labels(pred_labels, PUNCT_LABEL2ID, 'news'))