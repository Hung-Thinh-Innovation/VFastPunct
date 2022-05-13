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
        self.device = 'cuda' if not no_cuda and torch.cuda.is_available() else 'cpu'
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
        tokens = norm_text.split()
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
                if len(result) > 0 and result[-1] == ".":
                    result += f" {w.title()}{p}"
                else:
                    result += f" {w}{p}"
        return result.strip()


if __name__ == "__main__":
    from string import punctuation
    import re
    punct = VFastPunct("mBertLstmCrf", True)
    in_raw = 'Tại hội thảo, Trưởng Ban Tuyên giáo Trung ương Nguyễn Trọng Nghĩa nhấn mạnh trẻ em là nguồn hạnh phúc của ' \
             'gia đình, tương lai của dân tộc, lớp người kế tục sự nghiệp xây dựng và bảo vệ Tổ quốc. Do đó, tất cả trẻ ' \
             'em dưới 6 tuổi được cấp thẻ bảo hiểm y tế miễn phí; trẻ em dưới 1 tuổi được tham gia tiêm chủng mở rộng và ' \
             'trẻ em 5 tuổi được đi học mẫu giáo; học sinh tiểu học không phải trả phí đi học… Tuy vậy, ông Nghĩa chỉ rõ ' \
             'vấn đề mất cân bằng giới tính khi sinh, tình trạng tảo hôn, bạo lực gia đình còn gây nhiều hậu quả nghiêm ' \
             'trọng cho trẻ em. Tỉ lệ trẻ em dưới 5 tuổi suy dinh dưỡng thể thấp còi vẫn ở mức cao. Trẻ bị xâm hại, đặc ' \
             'biệt là xâm hại tình dục và bạo hành vẫn diễn biến phức tạp ở nhiều địa bàn, gây bức xúc trong xã hội.'.lower()
    in_text = re.sub(f" +", " ", re.sub(f"[{punctuation}]", " ", in_raw))
    print(punct(in_text))
