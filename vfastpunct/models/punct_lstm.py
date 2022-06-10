from vfastpunct.models.base_model import BaseModelOutput

from typing import Optional
from transformers import logging, BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings
from torchcrf import CRF

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.set_verbosity_error()


class PunctLstmConfig(BertConfig):
    def __init__(self, num_plabels=9, **kwargs):
        super().__init__(**kwargs, num_labels=num_plabels)
        self.num_plabels = num_plabels


class PunctBiLstmBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_plabels = config.num_plabels
        self.embeddings = BertEmbeddings(config)
        self.bilstm = nn.LSTM(input_size=config.hidden_size,
                              hidden_size=config.hidden_size // 2,
                              num_layers=2,
                              batch_first=True,
                              bidirectional=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.p_classifier = nn.Linear(config.hidden_size, config.num_plabels)

    @classmethod
    def from_pretrained(cls, model_name: str, config: PunctLstmConfig, from_tf: bool = False):
        model = cls(config)
        model.embeddings = BertModel.from_pretrained(model_name, config=config).embeddings
        return model

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None):
        return None


class PunctBiLstm(PunctBiLstmBase):
    def __init__(self, config):
        super(PunctBiLstm, self).__init__(config)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                plabels=None,
                clabels=None,
                valid_ids=None,
                label_masks=None):
        embedding_output =  self.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )
        seq_output, _ = self.bilstm(embedding_output)
        sequence_output = self.dropout(seq_output)
        p_logits = self.p_classifier(sequence_output)

        label_masks = label_masks.view(-1) != 0
        seq_ptags = torch.masked_select(torch.argmax(F.log_softmax(p_logits, dim=2), dim=2).view(-1), label_masks).tolist()
        if plabels is not None:
            p_loss = self.loss_func(p_logits.view(-1, self.num_plabels), plabels.view(-1))
            loss = p_loss
            return BaseModelOutput(loss=loss, ploss=p_loss, ptags=seq_ptags)
        else:
            return BaseModelOutput(ptags=seq_ptags)