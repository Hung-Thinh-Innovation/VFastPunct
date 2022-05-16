from vfastpunct.log import init_logger
from vfastpunct.models import PuncBertLstmCrf

LOGGER = init_logger()
EOS_MARKS = ["PERIOD", "QMARK", "EXCLAM"]
PUNC_LABEL2ID = ['O', 'PERIOD', 'COMMA', 'COLON', 'QMARK', 'EXCLAM', 'SEMICOLON', 'HYPHEN', 'ELLIPSIS']
PUNC_ID2LABEL = {idx: label for idx, label in enumerate(PUNC_LABEL2ID)}
CAP_LABEL2ID = ['KEEP', 'UPPER', 'TITLE']
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
    'KEEP': lambda x: x.lower(),
    'UPPER': lambda x: x.upper(),
    'TITLE': lambda x: x.title()
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
