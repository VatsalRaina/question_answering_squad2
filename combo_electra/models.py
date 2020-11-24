#! /usr/bin/env python

import torch
import torchvision.models as models
from transformers import ElectraModel, ElectraConfig

class ElectraQA(torch.nn.Module):
    def __init__(self):

        super(ElectraQA, self).__init__()

        self.electra = ElectraModel.from_pretrained("google/electra-base-discriminator")
        self.qa_outputs = torch.nn.Linear(self.electra.config.hidden_size, 2)
        self.classifier = torch.nn.Linear(self.electra.config.hidden_size, 1)
        self.dropout = torch.nn.Dropout(self.electra.config.hidden_dropout_prob)
        # self.init_weights()

    
    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

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

        outputs = self.electra(inputs_embeds=input_embeds)

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        verification_logits = torch.sigmoid(self.classifier(pooled_output))

        return start_logits, end_logits, verification_logits