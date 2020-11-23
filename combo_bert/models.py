#! /usr/bin/env python

import torch
import torchvision.models as models
from transformers import BertModel, BertConfig

class BertQA(torch.nn.Module):
    def __init__(self):

        super(BertQA, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.qa_outputs = torch.nn.Linear(self.bert.config.hidden_size, 2)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        # self.init_weights()

    
    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        verification_logits = torch.sigmoid(self.classifier(pooled_output))

        return start_logits, end_logits, verification_logits
    
    def saliency(self, input_embeds):

        outputs = self.bert(inputs_embeds=input_embeds)

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        verification_logits = torch.sigmoid(self.classifier(pooled_output))

        return start_logits, end_logits, verification_logits