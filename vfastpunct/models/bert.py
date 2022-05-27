from transformers import logging, BertForTokenClassification, BertConfig
from torchcrf import CRF

import torch
import torch.nn as nn

logging.set_verbosity_error()


class PuncCapBertConfig(BertConfig):
    def __init__(self, num_plabels=9, num_clabels=3, **kwargs):
        super().__init__(num_labels=num_plabels, **kwargs)
        self.num_plabels = num_plabels
        self.num_clabels = num_clabels


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

        seq_ptags = self.p_crf.decode(p_logits, mask=label_masks != 0)
        seq_ctags = self.c_crf.decode(c_logits, mask=label_masks != 0)

        if plabels is not None:
            p_log_likelihood = self.p_crf(p_logits, plabels, mask=label_masks.type(torch.uint8) != 0)
            c_log_likelihood = self.c_crf(c_logits, clabels, mask=label_masks.type(torch.uint8))
            loss = -1.0 * (p_log_likelihood+c_log_likelihood)
            return loss, p_log_likelihood, c_log_likelihood, seq_ptags, seq_ctags
        else:
            return seq_ptags, seq_ctags

class PuncBertLstmCrf(BertForTokenClassification):
    def __init__(self, config):
        super(PuncBertLstmCrf, self).__init__(config=config)
        self.num_labels = config.num_labels
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size // 2,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
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
        logits = self.classifier(sequence_output)
        seq_tags = self.crf.decode(logits, mask=label_masks != 0)
        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=label_masks.type(torch.uint8))
            return -1.0 * log_likelihood, seq_tags
        else:
            return seq_tags


# DEBUG
if __name__ == "__main__":
    from transformers import BertConfig
    model_name = 'bert-base-multilingual-cased'
    config = BertConfig.from_pretrained(model_name, num_labels=8)
    model = PuncBertLstmCrf.from_pretrained(model_name, config=config, from_tf=False)

    input_ids = torch.randint(0, 3000, [2, 20], dtype=torch.long)
    mask = torch.ones([2, 20], dtype=torch.long)
    labels = torch.randint(1, 7, [2, 20], dtype=torch.long)
    new_labels = torch.zeros([2, 20], dtype=torch.long)
    valid_ids = torch.ones([2, 20], dtype=torch.long)
    label_mask = torch.ones([2, 20], dtype=torch.long)
    valid_ids[:, 0] = 0
    valid_ids[:, 13] = 0
    labels[:, 0] = 0
    label_mask[:, -2:] = 0
    for i in range(len(labels)):
        idx = 0
        for j in range(len(labels[i])):
            if valid_ids[i][j] == 1:
                new_labels[i][idx] = labels[i][j]
                idx += 1
    output = model.forward(input_ids,
                   labels=new_labels,
                   attention_mask=mask,
                   valid_ids=valid_ids, label_masks=label_mask)
    print(labels)
    print(new_labels)
    print(label_mask)
    print(valid_ids)
    print(output)

