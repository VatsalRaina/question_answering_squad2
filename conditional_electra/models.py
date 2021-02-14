#! /usr/bin/env python

import torch
import torchvision.models as models
from transformers import ElectraModel, ElectraConfig

class ElectraClassificationHead(torch.nn.Module):
    def __init__(self):

        super(ElectraClassificationHead, self).__init__()

        electra_base = "google/electra-base-discriminator"
        electra_large = "google/electra-large-discriminator"
        self.electra = ElectraModel.from_pretrained(electra_large)
        self.dense = torch.nn.Linear(self.electra.config.hidden_size, self.electra.config.hidden_size)
        self.dropout = torch.nn.Dropout(self.electra.config.hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(self.electra.config.hidden_size, 1)
        self.gelu = torch.nn.GELU()

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.gelu(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ElectraQA(torch.nn.Module):
    def __init__(self):

        super(ElectraQA, self).__init__()

        electra_base = "google/electra-base-discriminator"
        electra_large = "google/electra-large-discriminator"
        self.electra = ElectraModel.from_pretrained(electra_large)
        self.qa_outputs = torch.nn.Linear(self.electra.config.hidden_size, 2)
        self.classifier = ElectraClassificationHead()
        self.dropout = torch.nn.Dropout(self.electra.config.hidden_dropout_prob)
        # self.init_weights()

    
    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        verification_logits = torch.sigmoid(self.classifier(sequence_output))

        return start_logits, end_logits, verification_logits
    
    def saliency(self, input_embeds):

        outputs = self.electra(inputs_embeds=input_embeds)

        sequence_output = outputs[0]
        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        verification_logits = torch.sigmoid(self.classifier(sequence_output))

        return start_logits, end_logits, verification_logits



class ElectraQAExtension(torch.nn.Module):
    def __init__(self, model_path, device):

        super(ElectraQAExtension, self).__init__()

        self.network = torch.load(model_path, map_location=device)
        for param in self.network.parameters():
            param.requires_grad = False
        # Transformer encoder to model the conditional probabilty (in both directions)
        # self.conditional_layer = torch.nn.TransformerEncoderLayer(d_model=2, nhead=1, dim_feedforward=128)
        self.conditional_layer = torch.nn.Linear(512, 512)


    
    def forward(self, input_ids, attention_mask, token_type_ids):

        start_logits, end_logits, verification_logits = self.network(input_ids, attention_mask, token_type_ids)


        # logits = torch.cat((torch.unsqueeze(start_logits,2), torch.unsqueeze(end_logits,2)), 2)
        # logits = torch.transpose(logits, 0, 1)
        # conditional_logits = self.conditional_layer(logits, src_key_padding_mask=~attention_mask.bool())
        # conditional_logits = torch.transpose(conditional_logits, 0, 1)
        # start_logits, end_logits = conditional_logits.split(1, dim=-1)
        # start_logits = start_logits.squeeze(-1)
        # end_logits = end_logits.squeeze(-1)

        # linear layer with residual connection from original end_logits
        end_logits = self.conditional_layer(torch.nn.GELU(start_logits)) + end_logits

        return start_logits, end_logits, verification_logits
    
    def saliency(self, input_embeds):

        outputs = self.electra(inputs_embeds=input_embeds)

        sequence_output = outputs[0]
        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        verification_logits = torch.sigmoid(self.classifier(sequence_output))

        return start_logits, end_logits, verification_logits