from vfastpunct.log import init_logger
from vfastpunct.models import (PuncCapLstmConfig, PuncCapBiLstm, PuncCapBiLstmCrf,
                               PuncCapBertConfig, PuncCapBert, PuncCapBertLstmCrf)


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

PUNCCAP_MODEL_MAPPING = {
    'lstm_crf': {
        'model_clss': PuncCapBiLstmCrf,
        'config_clss': PuncCapLstmConfig
    },
    'lstm_softmax': {
        'model_clss': PuncCapBiLstm,
        'config_clss': PuncCapLstmConfig
    },
    'bert_crf': {
        'model_clss': PuncCapBertLstmCrf,
        'config_clss': PuncCapBertConfig
    },
    'bert_softmax': {
        'model_clss': PuncCapBert,
        'config_clss': PuncCapBertConfig
    }
}

