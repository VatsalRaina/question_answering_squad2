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
from transformers import ElectraForQuestionAnswering, AdamW, ElectraConfig
from transformers import get_linear_schedule_with_warmup

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
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

def _find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1
    print("Didn't find match, return <no answer>")
    return 0,0


def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Choose device
    device = get_default_device()

    dev_data = load_dataset('squad_v2', split='validation')
    #print(dev_data[19])

    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', do_lower_case=True)


    start_positions_true = []
    end_positions_true = []
    labels = []
    input_ids = []
    token_type_ids = []
    count = 0
    no_answer = 0
    too_long = 0
    for ex in dev_data:
        count+=1
        #if count==1000:
        #    break
        question, passage = ex["question"], ex["context"]
        combo = question + " [SEP] " + passage
        inp_ids = tokenizer.encode(combo)
        if len(inp_ids) > 512:
            too_long+=1
            continue
            #print(ex)
        tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]  # Indicates whether part of sentence A or B -> 102 is Id of [SEP] token
        #print(ex)
        if len(ex["answers"]["text"])==0:
            start_idx, end_idx = 0, 0
            lab = 0
        else:
            ans_ids = tokenizer.encode(ex["answers"]["text"][0])
            ans_ids = ans_ids[1:-1]  # Remove the automatically added [CLS] and [SEP] tokens at beginning and end
            start_idx, end_idx = _find_sub_list(ans_ids, inp_ids)
            lab = 1
        if start_idx==0:
            no_answer+=1
        start_positions_true.append(start_idx)
        end_positions_true.append(end_idx)
        input_ids.append(inp_ids)
        token_type_ids.append(tok_type_ids)
        labels.append(lab)
    print("Greater than 512 tokens in length:")
    print(too_long)
    print(len(dev_data))
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
    start_positions_true = torch.tensor(start_positions_true)
    start_positions_true = start_positions_true.long().to(device)
    end_positions_true = torch.tensor(end_positions_true)
    end_positions_true = end_positions_true.long().to(device)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.long().to(device)
    attention_masks = torch.tensor(attention_masks)
    attention_masks = attention_masks.long().to(device)
    # Create the DataLoader for training set.
    ds = TensorDataset(input_ids, start_positions_true, end_positions_true, token_type_ids, attention_masks)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)
    pred_start_logits = []
    pred_end_logits = []
    count = 0
    for inp_id, s_pos, e_pos, tok_typ_id, att_msk in dl:
        print(count)
        count+=1
        inp_id, s_pos, e_pos, tok_typ_id, att_msk = inp_id.to(device), s_pos.to(device), e_pos.to(device), tok_typ_id.to(device), att_msk.to(device)
        with torch.no_grad():
            outputs = model(input_ids=inp_id, attention_mask=att_msk, token_type_ids=tok_typ_id, start_positions=s_pos, end_positions=e_pos)
        b_start_logits = outputs[1].detach().cpu().numpy().tolist()
        pred_start_logits += b_start_logits
        b_end_logits = outputs[2].detach().cpu().numpy().tolist()
        pred_end_logits += b_end_logits
    start_positions_true = start_positions_true.detach().cpu().numpy()
    end_positions_true = end_positions_true.detach().cpu().numpy()
    pred_start_logits, pred_end_logits = np.asarray(pred_start_logits), np.asarray(pred_end_logits)
    labels = np.asarray(labels)
    # Get the best values by looking at the maximum logits (it might be better to look at the best summed start and end logits)
    pred_start_pos = np.argmax(pred_start_logits, axis=1)
    pred_end_pos = np.argmax(pred_end_logits, axis=1)
    # Calculate the exact match now
    num_correct = 0
    num_correct_pos = 0
    num_correct_neg = 0
    total_num = len(start_positions_true)
    total_pos = np.sum(labels)
    total_neg = total_num - total_pos
    for i in range(total_num):
        if start_positions_true[i]==pred_start_pos[i] and end_positions_true[i]==pred_end_pos[i]:
            num_correct +=1
            if labels[i]==0:
                num_correct_neg += 1
            else:
                num_correct_pos += 1

    frac_correct = num_correct / total_num
    print("Fraction correct answers calculated: ")
    print(frac_correct)

    frac_correct_pos = num_correct_pos / total_pos
    print("Fraction correct answers calculated with label=1: ")
    print(frac_correct_pos)

    frac_correct_neg = num_correct_neg / total_neg
    print("Fraction correct answers calculated with label=0: ")
    print(frac_correct_neg)

    # Save the start and logits as well as the labels
    np.savetxt(args.predictions_save_path + "start_pos_true.txt", start_positions_true)
    np.savetxt(args.predictions_save_path + "end_pos_true.txt", end_positions_true)
    np.savetxt(args.predictions_save_path + "pred_start_logits.txt", pred_start_logits)
    np.savetxt(args.predictions_save_path + "pred_end_logits.txt", pred_end_logits)
    np.savetxt(args.predictions_save_path + "labels.txt", labels)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



