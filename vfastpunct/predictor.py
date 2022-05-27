from vfastpunct.constants import BASE_PATH, MODEL_MAPPING, PUNC_MAPPING, CAP_MAPPING, PUNC_LABEL2ID
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
        params = MODEL_MAPPING[model_name]
        model_path = os.path.join(BASE_PATH, f'{model_name.lower()}_{params["drive_id"]}.pt')
        if not os.path.exists(model_path):
            download_file_from_google_drive(params["drive_id"], model_path, confirm='t')
        self.model, self.tokenizer, self.max_seq_len, self.punc2id, self.cap2id = self.load_model(model_path=model_path,
                                                                                     device=self.device,
                                                                                     **params)
        self.id2puc = {idx: label for idx, label in enumerate(self.punc2id)}
        self.id2cap = {idx: label for idx, label in enumerate(self.cap2id)}

    @staticmethod
    def load_model(model_clss, config_clss, encode_name, model_path, device='cpu', **kwargs):
        if device == 'cpu':
            checkpoint_data = torch.load(model_path, map_location='cpu')
        else:
            checkpoint_data = torch.load(model_path)
        pclasses = PUNC_LABEL2ID
        cclasses = checkpoint_data.get('cclasses', [])
        if 'pclasses' not in checkpoint_data and 'classes' in checkpoint_data:
            pclasses = checkpoint_data['classes']
        max_seq_len = checkpoint_data['args'].max_seq_length
        tokenizer = AutoTokenizer.from_pretrained(encode_name)
        config = config_clss.from_pretrained(encode_name,
                                             num_plabels=len(checkpoint_data.get('pclasses', pclasses)),
                                             num_clabels=len(cclasses))
        model = model_clss(config=config)
        model.load_state_dict(checkpoint_data['model'])
        model.to(device)
        model.eval()
        return model, tokenizer, max_seq_len, pclasses, cclasses

    def _convert_tensor(self, sent):
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

    def _preprocess(self, in_raw: str):
        norm_text = normalize_text(in_raw)
        sents = []
        tokens = norm_text.split()
        idx = 0
        num_token = len(tokens)
        while num_token > idx >= 0:
            sents.append(' '.join(tokens[idx: min(idx + self.max_seq_len, num_token)]))
            idx += self.max_seq_len + 1
        return sents

    def _postprocess(self, sent, tags):
        result = ''
        if isinstance(tags, tuple):
            ptags, ctags = tags
            for w, ptag, ctag in list(zip(sent.split(), list(itertools.chain(*ptags)), list(itertools.chain(*ctags)))):
                p = PUNC_MAPPING[self.id2puc[ptag]]
                c = self.id2cap[ctag]
                if len(result) > 0 and result[-1] == ".":
                    result += f" {w.title()}{p}"
                else:
                    result += f" {CAP_MAPPING[c](w)}{p}"
            return result
        for w, l in list(zip(sent.split(), list(itertools.chain(*tags)))):
            p = PUNC_MAPPING[self.id2puc[l]]
            if len(result) > 0 and result[-1] == ".":
                result += f" {w.title()}{p}"
            else:
                result += f" {w}{p}"

    def __call__(self, in_raw: str):
        sents = self._preprocess(in_raw)
        result = ''
        for sent in sents:
            item = self._convert_tensor(sent)
            with torch.no_grad():
                tags = self.model(**item)
            result += self._postprocess(sent, tags)
        return result.strip()


if __name__ == "__main__":
    from string import punctuation
    import re
    punct = VFastPunct("mBertPuncCap", True)
    in_raw = 'việt nam quốc hiệu chính thức là cộng hòa xã hội chủ nghĩa việt nam là một quốc gia nằm ở cực đông của ' \
             'bán đảo đông dương thuộc khu vực đông nam á giáp với lào campuchia trung quốc biển đông và vịnh thái ' \
             'lan'.lower()
    in_text = re.sub(f" +", " ", re.sub(f"[{punctuation}]", " ", in_raw))
    print(punct(in_text))
