from transformers import BertForTokenClassification
from torchcrf import CRF

import torch
import torch.nn as nn


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

    def next_forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                     device='cuda'):
        seq_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        seq_output, _ = self.lstm(seq_output)
        batch_size, max_len, feat_dim = seq_output.shape
        sequence_output = self.dropout(seq_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=valid_ids.type(torch.uint8))
            return -1.0 * log_likelihood
        else:
            seq_tags = self.crf.decode(logits)
            return seq_tags

    def forward(self, token_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None):
        seq_output = self.bert(token_ids, token_type_ids, attention_mask, head_mask=None)[0]
        seq_output, _ = self.lstm(seq_output)
        sequence_output = self.dropout(seq_output)
        logits = self.classifier(sequence_output)
        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=valid_ids != 0)
            return -1.0 * log_likelihood
        else:
            seq_tags = self.crf.decode(logits)
            return seq_tags


if __name__ == "__main__":
    from transformers import BertConfig
    model_name = 'bert-base-multilingual-cased'
    config = BertConfig.from_pretrained(model_name, num_labels=8)
    model = PuncBertLstmCrf.from_pretrained(model_name, config=config, from_tf=False)

    input_ids = torch.randint(0, 3000, [1, 20], dtype=torch.long)
    mask = torch.ones([1, 20], dtype=torch.long)
    labels = torch.randint(0, 7, [1, 20], dtype=torch.long)
    valid_ids = torch.ones([1, 20], dtype=torch.long)
    label_mask = torch.ones([1, 20], dtype=torch.long)

    output = model(input_ids,
                   labels=labels,
                   attention_mask=mask,
                   valid_ids=valid_ids)
    print(output)

