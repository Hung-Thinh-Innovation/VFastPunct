from vfastpunct.log import init_logger
from vfastpunct.models import PuncBertLstmCrf

LOGGER = init_logger()
EOS_MARKS = ["PERIOD", "QMARK", "EXCLAM"]
PUNC_LABEL2ID = ['O', 'PERIOD', 'COMMA', 'COLON', 'QMARK', 'EXCLAM', 'SEMICOLON', 'HYPHEN', 'ELLIPSIS']
PUNC_ID2LABEL = {idx: label for idx, label in enumerate(PUNC_LABEL2ID)}
CAP_LABEL2ID = ['O', 'U', 'T']
CAP_ID2LABEL = {idx: label for idx, label in enumerate(CAP_LABEL2ID)}

PUNC_MAPPING = {
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
    'mBertLstmCrf': (PuncBertLstmCrf,
                     'bert-base-multilingual-cased',
                     '/tmp/bertlstmcrf.pt',
                     '17Ru3-tA98jcuV64rchf_zpfUj2K-TuZg')
}
