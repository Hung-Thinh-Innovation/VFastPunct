import os.path

from vfastpunct.constants import MODEL_MAPPING, PUNC_MAPPING
from vfastpunct.processor import normalize_text
from vfastpunct.ultis import download_file_from_google_drive
from transformers import AutoConfig, AutoTokenizer

import os
import torch
import itertools
import numpy as np


class VFastPunct(object):
    def __init__(self, model_name, no_cuda=False):
        self.device = device = 'cuda' if not no_cuda and torch.cuda.is_available() else 'cpu'
        model_clss,  lm_name, model_path, drive_id = MODEL_MAPPING[model_name]
        if not os.path.exists(model_path):
            download_file_from_google_drive(drive_id, model_path, confirm='t')
        self.model, self.tokenizer, self.max_seq_len, self.punc2id =self.load_model(model_clss,  lm_name, model_path, self.device)
        self.id2puc = {idx: label for idx, label in enumerate(self.punc2id)}
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def load_model(model_clss, lm_name, model_path, device='cpu'):
        if device == 'cpu':
            checkpoint_data = torch.load(model_path, map_location='cpu')
        else:
            checkpoint_data = torch.load(model_path)
        max_seq_len = checkpoint_data['args'].max_seq_length
        tokenizer = AutoTokenizer.from_pretrained(lm_name)
        config = AutoConfig.from_pretrained(lm_name, num_labels=len(checkpoint_data['classes']))
        model = model_clss(config=config)
        model.load_state_dict(checkpoint_data['model'])
        return model, tokenizer, max_seq_len, checkpoint_data['classes']

    def preprocess(self, in_raw: str):
        norm_text = normalize_text(in_raw)
        sents = []
        tokens = in_raw.split()
        idx = 0
        num_token = len(tokens)
        while num_token > idx >= 0:
            sents.append(' '.join(tokens[idx: min(idx + self.max_seq_len, num_token)]))
            idx += self.max_seq_len + 1
        return sents

    def convert_tensor(self, sent):
        encoding = self.tokenizer(normalize_text(sent),
                                  padding='max_length',
                                  truncation=True,
                                  return_offsets_mapping=True,
                                  max_length=self.max_seq_len)
        valid_id = np.zeros(len(encoding["offset_mapping"]), dtype=int)
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping == (0, 0) or (mapping[0] != 0 and mapping[0] == encoding["offset_mapping"][idx - 1][-1]):
                continue
            valid_id[idx] = 1
            i += 1
        label_masks = [1] * i
        label_masks.extend([0] * (self.max_seq_len - len(label_masks)))
        encoding.pop('offset_mapping', None)
        item = {key: torch.as_tensor([val]).to(self.device, dtype=torch.long) for key, val in encoding.items()}
        item['valid_ids'] = torch.as_tensor([valid_id]).to(self.device, dtype=torch.long)
        item['label_masks'] = torch.as_tensor([label_masks]).to(self.device, dtype=torch.long)
        return item

    def __call__(self, in_raw: str):
        sents = self.preprocess(in_raw)
        result = ''
        for sent in sents:
            item = self.convert_tensor(sent)
            with torch.no_grad():
                tag = self.model(**item)
            for w, l in list(zip(sent.split(), list(itertools.chain(*tag)))):
                p = PUNC_MAPPING[self.id2puc[l]]
                if p == ".":
                    result += f"{w.title()}{p} "
                else:
                    result += f"{w}{p} "
        return result.strip()


if __name__ == "__main__":
    from string import punctuation
    import re
    punct = VFastPunct("mBertLstmCrf", True)
    in_raw = 'Ngày 12-5, thông tin từ Công an tỉnh Long An cho biết trong ngày, Cơ quan an ninh điều tra Công an tỉnh ' \
              'này đã tống đạt quyết định khởi tố bị can, bắt tạm giam bà Cao Thị Cúc (62 tuổi, ngụ xã Hòa Khánh Tây, ' \
              'huyện Đức Hòa), là chủ căn nhà nơi tự xưng là "tịnh thất Bồng Lai".'.lower()
    in_text = re.sub(f" +", " ", re.sub(f"[{punctuation}]", " ", in_raw))
    print(punct(in_text))
