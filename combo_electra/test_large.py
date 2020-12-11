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
from keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


from models import ElectraQA

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
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
    # Force return cpu
    return torch.device('cpu')
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def _find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1
    print("Didn't find match, return <no answer>")
    return -1,0

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

    tokenizer = AutoTokenizer.from_pretrained("ahotrod/electra_large_discriminator_squad2_512")

    input_ids = []
    count = 0
    for ex in dev_data:
        count+=1
        if count==18:
           break
        question, passage = ex["question"], ex["context"]
        inputs = tokenizer.encode_plus(question, passage, add_special_tokens=True, return_tensors="pt")
        inp_ids = inputs["input_ids"].tolist()[0]
        if len(inp_ids) > 512:
            inp_ids = inp_ids[:512]
        input_ids.append(inp_ids)
    # Pad all sequences with 0
    max_len = max([len(sen) for sen in input_ids])
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    # Convert to torch tensors
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    # Create the DataLoader for test set.
    ds = TensorDataset(input_ids)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = AutoModelForQuestionAnswering.from_pretrained("ahotrod/electra_large_discriminator_squad2_512")
    model.eval().to(device)
    pred_start_logits = []
    pred_end_logits = []
    count = 0
    for item in dl:
        print(count)
        count+=1
        inp_id = item[0].to(device)
        with torch.no_grad():
            start_logits, end_logits = model(inp_id)
        print(start_logits.shape)
        b_start_logits = torch.squeeze(start_logits).detach().cpu().numpy().tolist()
        # pred_start_logits += b_start_logits
        b_end_logits = torch.squeeze(end_logits).detach().cpu().numpy().tolist()
        if len(pred_start_logits)==0:
            pred_start_logits += b_start_logits
            pred_end_logits += b_end_logits
        else:
            pred_start_logits.extend(b_start_logits)
            pred_end_logits.extend(b_end_logits)
        # pred_end_logits += b_end_logits
    pred_start_logits, pred_end_logits = np.asarray(pred_start_logits), np.asarray(pred_end_logits)
    # Save all necessary file (in order to be able to ensemble)
    np.savetxt(args.predictions_save_path + "pred_start_logits_all.txt", pred_start_logits)
    np.savetxt(args.predictions_save_path + "pred_end_logits_all.txt", pred_end_logits)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)