#! /usr/bin/env python

import argparse
import os
import sys

import torch
import numpy as np
from scipy.stats import entropy

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


from models import ElectraQA

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
parser.add_argument('--predictions_save_path', type=str, help="Where to save predicted values")


def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    dev_data = load_dataset('squad_v2', split='validation')
    print(len(dev_data))

    tokenizer = AutoTokenizer.from_pretrained("ahotrod/electra_large_discriminator_squad2_512")
    model = AutoModelForQuestionAnswering.from_pretrained("ahotrod/electra_large_discriminator_squad2_512")
    count = 0
    span_predictions = {}
    entropy_on = []
    entropy_off = []
    for ex in dev_data:
        count+=1
        print(count)
        if count==10:
           break
        question, passage, qid = ex["question"], ex["context"], ex["id"]
        inputs = tokenizer.encode_plus(question, passage, add_special_tokens=True, return_tensors="pt")
        inp_ids = inputs["input_ids"].tolist()[0]
 
        start_logits, end_logits = model(**inputs)
        answer_start = torch.argmax(start_logits)
        answer_end = torch.argmax(end_logits)
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inp_ids[answer_start:answer_end+1]))
        
        span_predictions[qid] = answer

        start_logits = start_logits.detach().cpu().numpy()
        end_logits = end_logits.detach().cpu().numpy()

        start_probs = start_logits / np.sum(start_logits)
        end_probs = end_logits / np.sum(end_logits)

        entrop = (entropy(start_probs, base=2) + entropy(end_probs, base=2)) / 2

        if len(ex["answers"]["text"])==0:
            entropy_off.append(entrop)
        else:
            entropy_on.append(entrop)
            # print(question)
            # print(passage)
            # print(ex["answers"]["text"])
            # print(answer)
    print(entropy_on)
    print(entropy_off)


    # pred_start_logits = []
    # pred_end_logits = []
    # count = 0
    # for item in dl:
    #     print(count)
    #     count+=1
    #     inp_id = item[0].to(device)
    #     with torch.no_grad():
    #         start_logits, end_logits = model(inp_id)
    #     b_start_logits = start_logits.detach().cpu().numpy().tolist()
    #     # pred_start_logits += b_start_logits
    #     b_end_logits = end_logits.detach().cpu().numpy().tolist()
    #     if len(pred_start_logits)==0:
    #         pred_start_logits += b_start_logits
    #         pred_end_logits += b_end_logits
    #     else:
    #         pred_start_logits.extend(b_start_logits)
    #         pred_end_logits.extend(b_end_logits)
    #     # pred_end_logits += b_end_logits
    # pred_start_logits, pred_end_logits = np.asarray(pred_start_logits), np.asarray(pred_end_logits)
    # Save all necessary file (in order to be able to ensemble)
    # np.savetxt(args.predictions_save_path + "pred_start_logits_all.txt", pred_start_logits)
    # np.savetxt(args.predictions_save_path + "pred_end_logits_all.txt", pred_end_logits)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)