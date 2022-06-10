from vfastpunct.models.base_model import BaseModelOutput

from transformers import logging, BertForTokenClassification, BertConfig
from torchcrf import CRF

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

logging.set_verbosity_error()


class PuncCapBertConfig(BertConfig):
    def __init__(self, num_plabels=9, num_clabels=3, **kwargs):
        super().__init__(num_labels=num_plabels, **kwargs)
        self.num_plabels = num_plabels
        self.num_clabels = num_clabels


class PunctCapBert(BertForTokenClassification):
    def __init__(self, config):
        super(PunctCapBert, self).__init__()
        self.num_plabels = config.num_plabels
        self.num_clabels = config.num_clabels
        self.classifier = nn.Linear(config.hidden_size, config.num_plabels)
        self.c_classifier = nn.Linear(config.hidden_size, config.num_clabels)
        self.post_init()

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                plabels=None,
                clabels=None,
                valid_ids=None,
                label_masks=None):
        seq_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        sequence_output = self.dropout(seq_output)

        p_logits = self.classifier(sequence_output)
        c_logits = self.c_classifier(sequence_output)
        label_masks = label_masks.view(-1) != 0
        seq_ptags = torch.masked_select(torch.argmax(F.log_softmax(p_logits, dim=2), dim=2).view(-1),
                                        label_masks).tolist()
        seq_ctags = torch.masked_select(torch.argmax(F.log_softmax(c_logits, dim=2), dim=2).view(-1),
                                        label_masks).tolist()
        if plabels is not None:
            ploss = self.loss_func(p_logits.view(-1, self.num_plabels), plabels.view(-1))
            closs = self.loss_func(c_logits.view(-1, self.num_clabels), clabels.view(-1))
            loss = ploss + closs
            return BaseModelOutput(loss=loss, ploss=ploss, closs=closs, ptags=seq_ptags, ctags=seq_ctags)
        else:
            return BaseModelOutput(ptags=seq_ptags, ctags=seq_ctags)


class PuncCapBertLstmCrf(BertForTokenClassification):
    def __init__(self, config):
        super(PuncCapBertLstmCrf, self).__init__(config=config)
        self.num_labels = config.num_labels
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size // 2,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.p_classifier = nn.Linear(config.hidden_size, config.num_plabels)
        self.c_classifier = nn.Linear(config.hidden_size, config.num_clabels)
        self.p_crf = CRF(config.num_plabels, batch_first=True)
        self.c_crf = CRF(config.num_clabels, batch_first=True)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                plabels=None,
                clabels=None,
                valid_ids=None,
                label_masks=None):
        seq_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        seq_output, _ = self.lstm(seq_output)

        batch_size, max_len, feat_dim = seq_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=seq_output.device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = seq_output[i][j]

        sequence_output = self.dropout(valid_output)

        p_logits = self.p_classifier(sequence_output)
        c_logits = self.c_classifier(sequence_output)

        seq_ptags = list(itertools.chain(*self.p_crf.decode(p_logits, mask=label_masks != 0)))
        seq_ctags = list(itertools.chain(*self.c_crf.decode(c_logits, mask=label_masks != 0)))
        if plabels is not None:
            p_log_likelihood = self.p_crf(p_logits, plabels, mask=label_masks.type(torch.uint8) != 0)
            c_log_likelihood = self.c_crf(c_logits, clabels, mask=label_masks.type(torch.uint8))
            loss = -1.0 * (p_log_likelihood + c_log_likelihood)
            ploss = -1.0 * (p_log_likelihood)
            closs = -1.0 * (c_log_likelihood)
            return BaseModelOutput(loss=loss, ploss=ploss, closs=closs, ptags=seq_ptags, ctags=seq_ctags)
        else:
            return BaseModelOutput(ptags=seq_ptags, ctags=seq_ctags)


# DEBUG
if __name__ == "__main__":
    from transformers import BertConfig

    model_name = 'bert-base-multilingual-cased'
    config = PuncCapBertConfig.from_pretrained(model_name, num_plabels=9, num_clabels=3)
    model = PuncCapBertLstmCrf.from_pretrained(model_name, config=config, from_tf=False)

    input_ids = torch.randint(0, 3000, [2, 20], dtype=torch.long)
    mask = torch.ones([2, 20], dtype=torch.long)
    plabels = torch.randint(0, 8, [2, 20], dtype=torch.long)
    clabels = torch.randint(0, 2, [2, 20], dtype=torch.long)
    new_plabels = torch.zeros([2, 20], dtype=torch.long)
    new_clabels = torch.zeros([2, 20], dtype=torch.long)
    valid_ids = torch.ones([2, 20], dtype=torch.long)
    label_mask = torch.ones([2, 20], dtype=torch.long)
    valid_ids[:, 0] = 0
    valid_ids[:, 13] = 0
    plabels[:, 0] = 0
    clabels[:, 0] = 0
    label_mask[:, -2:] = 0
    for i in range(len(plabels)):
        idx = 0
        for j in range(len(plabels[i])):
            if valid_ids[i][j] == 1:
                new_plabels[i][idx] = plabels[i][j]
                idx += 1
    for i in range(len(clabels)):
        idx = 0
        for j in range(len(clabels[i])):
            if valid_ids[i][j] == 1:
                new_clabels[i][idx] = clabels[i][j]
                idx += 1
    output = model.forward(input_ids,
                           plabels=new_plabels,
                           clabels=new_clabels,
                           attention_mask=mask,
                           valid_ids=valid_ids, label_masks=label_mask)
    print(plabels)
    print(clabels)
    print(new_plabels)
    print(new_clabels)
    print(label_mask)
    print(valid_ids)
    print(output)