from vfastpunct.log import init_logger
from vfastpunct.models import *
from datetime import datetime
from string import punctuation

import re


LOGGER = init_logger(datetime.now().strftime('%d%b%Y_%H-%M-%S.log'))
EOS_MARKS = ["PERIOD", "QMARK", "EXCLAM"]
RATE_MAKS = []
PUNCT_LABEL2ID = ['O', 'PERIOD', 'COMMA', 'COLON', 'QMARK', 'EXCLAM', 'SEMICOLON', 'HYPHEN', 'ELLIPSIS']
PUNCT_ID2LABEL = {idx: label for idx, label in enumerate(PUNCT_LABEL2ID)}
CAP_LABEL2ID = ['O', 'U', 'T']
CAP_ID2LABEL = {idx: label for idx, label in enumerate(CAP_LABEL2ID)}

PUBLIC_PUNCT_LABEL2ID = ['O', 'PERIOD', 'COMMA', 'COLON', 'QMARK', 'EXCLAM', 'SEMICOLON']
PUBLIC_PUNCT_ID2LABEL = {idx: label for idx, label in enumerate(PUBLIC_PUNCT_LABEL2ID)}

PUNCT_PATTERN = re.compile(f'[{punctuation}]+')
CAP_PATTERN = re.compile(f"[A-Z|ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸ]").search

PUNCT_MAPPING = {
    'PERIOD': '.',
    'COMMA': ',',
    'COLON': ':',
    'QMARK': '?',
    'EXCLAM': '!',
    'SEMICOLON': ';',
    'HYPHEN': ' -',
    'ELLIPSIS': '...',
    'O': ''
}

CAP_MAPPING = {
    'O': lambda x: x.lower(),
    'U': lambda x: x.upper(),
    'T': lambda x: x.title()
}

STRPUNC_MAPPING = {
    '.': 'PERIOD',
    ',': 'COMMA',
    ':': 'COLON',
    '?': 'QMARK',
    '!': 'EXCLAM',
    ';': 'SEMICOLON',
    '-': 'HYPHEN',
    '...': 'ELLIPSIS'
}

MODEL_MAPPING = {
    'punct': {
        'lstm_crf': {
            'model_clss': None,
            'config_clss': PunctLstmConfig
        },
        'lstm_softmax': {
            'model_clss': PunctBiLstm,
            'config_clss': PunctLstmConfig
        },
        'bert_crf': {
            'model_clss': None,
            'config_clss': None
        },
        'bert_softmax': {
            'model_clss': None,
            'config_clss': None
        }
    },
    'punctcap': {
        'lstm_crf': {
            'model_clss': PunctCapBiLstmCrf,
            'config_clss': PunctCapLstmConfig
        },
        'lstm_softmax': {
            'model_clss': PunctCapBiLstm,
            'config_clss': PunctCapLstmConfig
        },
        'bert_crf': {
            'model_clss': PuncCapBertLstmCrf,
            'config_clss': PuncCapBertConfig
        },
        'bert_softmax': {
            'model_clss': PunctCapBert,
            'config_clss': PuncCapBertConfig
        }
    }
}

DATA_SOURCES = {
    "train_016_splitted.txt": '1sdX1WIXo0uIEg7ZT5PqloJgrIbE8ulOC',
    "train_015_splitted.txt": '1WQupNn0tRPdAKu3jBP9CEaXNx8fqWART',
    "train_014_splitted.txt": '1FEyk15PGOX-Ue3qdgV0LRBYOnxbBe9GQ',
    "train_013_splitted.txt": '1q5CZPoOeOSS6sa78zCs0L1thQNhxxkfn',
    "train_012_splitted.txt": '1NIXqZai3KtdgPqk2cciUOVq9dIlmBFuj',
    "train_011_splitted.txt": '1Hz5pULnsLwgXe-mDnstVBreE5fh8rFIj',
    "train_010_splitted.txt": '1YOnoxs1otSh678nTNfhwOJOef7oBXzgW',
    "train_009_splitted.txt": '1HfxUYJ7g1QjIX6tA3yiSNrSJ2DIRc4LF',
    "train_008_splitted.txt": '1_SD5RQ-_OYVCR9W_Sdo-q_xXekRcWk7g',
    "train_007_splitted.txt": '1IR1OvKi3wFe2YmzjPkO0oCrXXIpeUxON',
    "train_006_splitted.txt": '1P86W3ZdkgK-occt1k8QwBOjwf0CxD5_q',
    "train_005_splitted.txt": '1YlfkusvoKfswnPyZyIVHC7I1xXdvBnXo',
    "train_004_splitted.txt": '1QryziNLPTma2q_Hf9PIjiQltT_akZXSf',
    "train_003_splitted.txt": '1XtaioCuKX-w2fvSWGkjNAnbHlsNeYdDu',
    "train_002_splitted.txt": '16GSQEN2HhnMUxx3cYfg52VrKGzxs9YdW',
    "train_001_splitted.txt": '1mjMdIxt6xa3ayfdAexR_PjeQprH8Zn5q',
    "train_000_splitted.txt": '1jSTjRwPMbvy8SDcEy_fHQqtkz8fiZtaR',
    "test_000_splitted.txt": '1bUGX-PxJ2pI-Ov9UGVii8PrYuxDW3aNH',
}

