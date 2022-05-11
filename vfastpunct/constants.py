from vfastpunct.log import init_logger
from vfastpunct.models import PuncBertLstmCrf

LOGGER = init_logger()
EOS_MARKS = ["PERIOD", "QMARK", "EXCLAM"]
PUNC_LABEL2ID = ['O', 'PERIOD', 'COMMA', 'COLON', 'QMARK', 'EXCLAM', 'SEMICOLON']
PUNC_ID2LABEL = {idx: label for idx, label in enumerate(PUNC_LABEL2ID)}

MODEL_MAPPING = {
    'BertLstmCrf-cased': (PuncBertLstmCrf,)
}