from vfastpunct.log import init_logger
from vfastpunct.models import PuncBertLstmCrf

LOGGER = init_logger()
EOS_MARKS = ["PERIOD", "QMARK", "EXCLAM"]
PUNC_LABEL2ID = ['O', 'PERIOD', 'COMMA', 'COLON', 'QMARK', 'EXCLAM', 'SEMICOLON']
PUNC_ID2LABEL = {idx: label for idx, label in enumerate(PUNC_LABEL2ID)}

PUNC_MAPPING = {'PERIOD': '.',
                'COMMA': ',',
                'COLON': ':',
                'QMARK': '?',
                'EXCLAM': '!',
                'SEMICOLON': ';',
                'O': ''
}

MODEL_MAPPING = {
    'mBertLstmCrf': (PuncBertLstmCrf,
                     'bert-base-multilingual-cased',
                     '/tmp/bertlstmcrf.pt',
                     '17Ru3-tA98jcuV64rchf_zpfUj2K-TuZg')
}