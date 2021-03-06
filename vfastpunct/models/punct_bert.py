from vfastpunct.models.base_model import BaseModelOutput

from transformers import logging, BertForTokenClassification
from torchcrf import CRF

import torch
import torch.nn as nn

logging.set_verbosity_error()


class PuncBertLstmCrf(BertForTokenClassification):
    def __init__(self, config):
        super(PuncBertLstmCrf, self).__init__(config=config)
        self.num_labels = config.num_plabels
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size // 2,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(config.num_plabels, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                label_masks=None) -> BaseModelOutput:
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
            loss = -1.0 * log_likelihood
            return BaseModelOutput(loss=loss, ploss=loss, ptags=seq_tags)
        else:
            return BaseModelOutput(ptags=seq_tags)


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