import os

from vfastpunct.constants import LOGGER, DATA_SOURCES

import requests


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