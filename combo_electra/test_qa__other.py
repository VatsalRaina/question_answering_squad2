#! /usr/bin/env python

import argparse
import os
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import datetime

from datasets import load_dataset
from transformers import ElectraTokenizer
from keras.preprocessing.sequence import pad_sequences
from transformers import AdamW, ElectraConfig

from models import ElectraQA

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the evaluation batch size')
parser.add_argument('--prompts_path', type=str, help='Load path prompts')
parser.add_argument('--responses_path', type=str, help='Load path responses')
parser.add_argument('--model_path', type=str, help='Load path to trained model')
parser.add_argument('--predictions_save_path', type=str, help="Where to save predicted values")


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Choose device
    device = get_default_device()

    dev_data = load_dataset('squad_v2', split='validation')
    print(len(dev_data))

    electra_base = "google/electra-base-discriminator"
    electra_large = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)

    with open(args.prompts_path) as f:
        prompts = f.readlines()
    # Remove whitespaces and convert to lower case
    prompts = [x.strip().lower() for x in prompts]

    with open(args.responses_path) as f:
        responses = f.readlines()
    # Remove whitespaces and convert to lower case
    responses = [x.strip().lower() for x in responses]

    input_ids = []
    token_type_ids = []
    count = 0
    for prompt, response in zip(prompts, responses):
        count+=1
        # if count==18:
        #    break
        question, passage = prompt, response
        combo = question + " [SEP] " + passage
        inp_ids = tokenizer.encode(combo)
        if len(inp_ids) > 512:
            inp_ids = inp_ids[:512]
        tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]  # Indicates whether part of sentence A or B -> 102 is Id of [SEP] token
        input_ids.append(inp_ids)
        token_type_ids.append(tok_type_ids)
    # Pad all sequences with 0
    max_len = max([len(sen) for sen in input_ids])
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    token_type_ids = pad_sequences(token_type_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    # Create attention masks
    attention_masks = []
    for sen in input_ids:
        att_mask = [int(token_id > 0) for token_id in sen]
        attention_masks.append(att_mask)
    # Convert to torch tensors
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.long().to(device)
    attention_masks = torch.tensor(attention_masks)
    attention_masks = attention_masks.long().to(device)
    # Create the DataLoader for training set.
    ds = TensorDataset(input_ids, token_type_ids, attention_masks)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)
    pred_start_logits = []
    pred_end_logits = []
    pred_verification_logits = []
    count = 0
    for inp_id, tok_typ_id, att_msk in dl:
        print(count)
        count+=1
        inp_id, tok_typ_id, att_msk = inp_id.to(device), tok_typ_id.to(device), att_msk.to(device)
        with torch.no_grad():
            start_logits, end_logits, verification_logits = model(input_ids=inp_id, attention_mask=att_msk, token_type_ids=tok_typ_id)
        b_start_logits = start_logits.detach().cpu().numpy().tolist()
        pred_start_logits += b_start_logits
        b_end_logits = end_logits.detach().cpu().numpy().tolist()
        pred_end_logits += b_end_logits
        b_verification_logits = verification_logits.detach().cpu().numpy().tolist()
        pred_verification_logits += b_verification_logits
    pred_start_logits, pred_end_logits = np.asarray(pred_start_logits), np.asarray(pred_end_logits)
    pred_verification_logits = np.asarray(pred_verification_logits)


    np.save(args.predictions_save_path + "pred_start_logits_all.npy", pred_start_logits)
    np.save(args.predictions_save_path + "pred_end_logits_all.npy", pred_end_logits)
    np.save(args.predictions_save_path + "verification_probs_all.npy", pred_verification_logits)

    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)