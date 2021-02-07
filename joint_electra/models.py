#! /usr/bin/env python

import torch
import torchvision.models as models
from transformers import ElectraModel, ElectraConfig

class ElectraQA(torch.nn.Module):
    def __init__(self):

        super(ElectraQA, self).__init__()

        electra_base = "google/electra-base-discriminator"
        electra_large = "google/electra-large-discriminator"
        self.electra = ElectraModel.from_pretrained(electra_large)
        L_max = 512
        self.qa_outputs = torch.nn.Linear(self.electra.config.hidden_size, L_max)
        # self.init_weights()

    
    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)

        logits = self.qa_outputs(sequence_output)

        return torch.reshape(logits,(logits.size(0),-1))
