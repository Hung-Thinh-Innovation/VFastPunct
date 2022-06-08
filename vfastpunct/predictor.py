from vfastpunct.constants import MODEL_MAPPING, PUNCT_MAPPING, CAP_MAPPING
from vfastpunct.processor import normalize_text
from vfastpunct.ultis import download_file_from_google_drive

from transformers import AutoConfig, AutoTokenizer

import os
import torch
import itertools
import numpy as np


class VFastPunct(object):
    def __init__(self, model_path, no_cuda=False):
        self.device = 'cuda' if not no_cuda and torch.cuda.is_available() else 'cpu'
        self.model, self.tokenizer, self.max_seq_len, self.punc2id, self.cap2id = self.load_model(model_path=model_path,
                                                                                                  device=self.device)
        self.id2puc = {idx: label for idx, label in enumerate(self.punc2id)}
        self.id2cap = {idx: label for idx, label in enumerate(self.cap2id)}

    @staticmethod
    def load_model(model_path, device='cpu'):
        if device == 'cpu':
            checkpoint_data = torch.load(model_path, map_location='cpu')
        else:
            checkpoint_data = torch.load(model_path)
        configs = checkpoint_data['args']
        tokenizer = AutoTokenizer.from_pretrained(configs.model_name_or_path)
        model_archs = MODEL_MAPPING['punctcap'][configs.model_arch]
        pclss = checkpoint_data['pclasses']
        cclss = checkpoint_data['cclasses']
        config = model_archs['config_clss'].from_pretrained(configs.model_name_or_path,
                                                            num_plabels=len(pclss),
                                                            num_clabels=len(cclss),
                                                            finetuning_task=configs.task)
        max_seq_len = checkpoint_data['args'].max_seq_length
        model = model_archs['model_clss'](config=config)
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(checkpoint_data['model'])
        model.to(device)
        model.eval()
        return model, tokenizer, max_seq_len, pclss, cclss

    def _convert_tensor(self, sent):
        encoding = self.tokenizer(normalize_text(sent),
                                  padding='max_length',
                                  truncation=True,
                                  return_offsets_mapping=True,
                                  max_length=self.max_seq_len)
        valid_id = np.ones(len(encoding["offset_mapping"]), dtype=int)
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping == (0, 0) or (mapping[0] != 0 and mapping[0] == encoding["offset_mapping"][idx - 1][-1]):
                valid_id[idx] = 0
                continue
            i += 1
        label_masks = [1] * i
        label_masks.extend([0] * (self.max_seq_len - len(label_masks)))
        encoding.pop('offset_mapping', None)
        item = {key: torch.as_tensor([val]).to(self.device, dtype=torch.long) for key, val in encoding.items()}
        item['valid_ids'] = torch.as_tensor([valid_id]).to(self.device, dtype=torch.long)
        item['label_masks'] = torch.as_tensor([valid_id]).to(self.device, dtype=torch.long)
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
            for w, ptag, ctag in list(zip(sent.split(), ptags, ctags)):
                p = PUNCT_MAPPING[self.id2puc[ptag]]
                c = self.id2cap[ctag]
                if len(result) > 0 and result[-1] == ".":
                    result += f" {w.title()}{p}"
                else:
                    result += f" {CAP_MAPPING[c](w)}{p}"
            return result
        for w, l in list(zip(sent.split(), list(itertools.chain(*tags)))):
            p = PUNCT_MAPPING[self.id2puc[l]]
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
                outputs = self.model(**item)
            result += self._postprocess(sent, (outputs.ptags, outputs.ctags))
        return result.strip()


if __name__ == "__main__":
    from string import punctuation
    import re
    punct = VFastPunct("./best_model.pt", True)
    while True:
        in_raw = input()
        # in_raw = 'việt nam quốc hiệu chính thức là cộng hòa xã hội chủ nghĩa việt nam là một quốc gia nằm ở cực đông của ' \
        #          'bán đảo đông dương thuộc khu vực đông nam á giáp với lào campuchia trung quốc biển đông và vịnh thái ' \
        #          'lan'.lower()
        in_text = re.sub(f" +", " ", re.sub(f"[{punctuation}]", " ", in_raw).lower())
        print(punct(in_text))