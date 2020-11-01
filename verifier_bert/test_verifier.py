#! /usr/bin/env python

import argparse
import os
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import precision_recall_curve
import random
import time
import datetime

from datasets import load_dataset
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from transformers import BertForSequenceClassification


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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    input_ids = []
    token_type_ids = []
    labels = []
    count = 0
    too_long = 0
    for ex in dev_data:
        count+=1
        # if count==17:
        #     break
        question, passage = ex["question"], ex["context"]
        combo = question + " [SEP] " + passage
        inp_ids = tokenizer.encode(combo)
        if len(inp_ids) > 512:
            too_long+=1
            continue
        tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]  # Indicates whether part of sentence A or B -> 102 is Id of [SEP] token
        if len(ex["answers"]["text"])==0:
            lab = 0
        else:
            lab = 1
        input_ids.append(inp_ids)
        token_type_ids.append(tok_type_ids)
        labels.append(lab)
    print("Greater than 512 tokens in length:")
    print(too_long)
    # Pad all sequences with 0
    max_len = max([len(sen) for sen in input_ids])
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    token_type_ids = pad_sequences(token_type_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    # Create attention masks
    attention_masks = []
    for sen in input_ids:
        att_mask = [int(token_id > 0) for token_id in sen]
        attention_masks.append(att_mask)
    #  Convert to torch tensors
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.long().to(device)
    attention_masks = torch.tensor(attention_masks)
    attention_masks = attention_masks.long().to(device)
    labels = torch.tensor(labels)
    labels = labels.long().to(device)

    ds = TensorDataset(input_ids, token_type_ids, attention_masks, labels)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = torch.load(args.model_path)
    model.eval().to(device)

    y_pred_all = []
    for b_input_ids, b_tok_typ_ids, b_att_msks, b_labels in dl:
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=b_tok_typ_ids, attention_mask=b_att_msks, labels=b_labels)
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        logits = np.squeeze(logits[:, 1])
        logits = logits.tolist()
        y_pred_all += logits
    y_pred_all = np.array(y_pred_all)
    y_true = labels.detach().cpu().numpy()

    # Save the predicted values so that they can be used for ensembling
    np.savetxt(args.predictions_save_path, y_pred_all)

    # Calculate and report best F0.5 score
    # Label 1 indicates on-topic and 0 indicates off-topic
    # But we care about detecting off-topic data, so to use F0.5, we do 1. - all values
    y_true = 1.-y_true
    y_pred = 1.-y_pred_all
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    f_score = np.amax( (1.+0.5**2) * ( (precision * recall) / (0.5**2 * precision + recall) ) )
    print("F0.5 score is:")
    print(f_score)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args) 