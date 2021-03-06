from vfastpunct.log import init_logger
from vfastpunct.models import PuncBertLstmCrf, PuncCapBertLstmCrf, PuncCapBertConfig, PunctCapBiLstm, PunctCapLstmConfig


LOGGER = init_logger()
# PUNC_LABEL2ID = ['O', 'PERIOD', 'COMMA', 'COLON', 'QMARK', 'EXCLAM', 'SEMICOLON']
EOS_MARKS = ["PERIOD", "QMARK", "EXCLAM"]
PUNC_LABEL2ID = ['O', 'PERIOD', 'COMMA', 'COLON', 'QMARK', 'EXCLAM', 'SEMICOLON', 'HYPHEN', 'ELLIPSIS']
CAP_LABEL2ID = ['O', 'U', 'T']
PUNC_ID2LABEL = {idx: label for idx, label in enumerate(PUNC_LABEL2ID)}
CAP_ID2LABEL = {idx: label for idx, label in enumerate(CAP_LABEL2ID)}


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


MODEL_MAPPING = {
    'mBertPunct': {
        'model_clss': PuncBertLstmCrf,
        'config_clss': PuncCapBertConfig,
        'encode_name': 'bert-base-multilingual-cased',
        'drive_id': '17Ru3-tA98jcuV64rchf_zpfUj2K-TuZg',
        'md5': '24f9324f211ed6bcc0c16b024023c142',
    },
    'mBertPunctCap': {
        'model_clss': PuncCapBertLstmCrf,
        'config_clss': PuncCapBertConfig,
        'encode_name': 'bert-base-multilingual-cased',
        'drive_id': '1Iv3iQfuA7NWRa2lQgWVn4fMLk4-XkWwZ',
        'md5': '958d822986892596b846f4e9b7316a64',
    },
    'mLstmPunctCap': {
        'model_clss': PunctCapBiLstm,
        'config_clss': PunctCapLstmConfig,
        'encode_name': 'bert-base-multilingual-cased',
        'drive_id': '11tsglNoAJzpCwAkWXs8OGIo2WJCqPw59',
        'md5': 'cc3348c89d6d735484e14f71ccdf459a',
    },
}


BASE_PATH = '/tmp/'