from vfastpunct.constants import MODEL_MAPPING


class VFastPunct(object):
    def __init__(self, model_name):
        model_clss = MODEL_MAPPING[model_name]

    def __call__(self, in_text: str):
        return in_text
