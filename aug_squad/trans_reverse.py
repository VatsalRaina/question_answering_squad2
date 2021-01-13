#! /usr/bin/env python

import argparse
import os
import sys
import json

import torch
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--predictions_save_path', type=str, help="Where to save augmented and original data")

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    train_data = load_dataset('squad_v2', split='train')
    print(len(train_data))

    tokenizer_en_to_fr = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    tokenizer_fr_to_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

    model_en_to_fr = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    model_fr_to_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

    aug_train = {}
    previous_passage = ""
    previous_aug_passage = ""
    count = 0
    for ex in train_data:
        count+=1
        print(count)
        if count==20001:
           break
        question, passage, qid = ex["question"], ex["context"], ex["id"]
        answers = ex["answers"]["text"]

        current = {}
        current['question'] = question
        current['context'] = passage
        current['answers'] = answers

        if passage == previous_passage:
            current['aug_context'] = previous_aug_passage
        else:
            # Tokenize the text
            batch = tokenizer_en_to_fr.prepare_seq2seq_batch(src_texts=[passage])
            # Make sure that the tokenized text does not exceed the maximum
            # allowed size of 512
            batch["input_ids"] = batch["input_ids"][:, :512]
            batch["attention_mask"] = batch["attention_mask"][:, :512]

            # Perform the translation and decode the output
            translation = model_en_to_fr.generate(**batch)
            french = tokenizer_en_to_fr.batch_decode(translation, skip_special_tokens=True)[0]

            # Now perform the reverse translation

            # Tokenize the text
            batch = tokenizer_fr_to_en.prepare_seq2seq_batch(src_texts=[french])
            # Make sure that the tokenized text does not exceed the maximum
            # allowed size of 512
            batch["input_ids"] = batch["input_ids"][:, :512]
            batch["attention_mask"] = batch["attention_mask"][:, :512]

            # Perform the translation and decode the output
            translation = model_fr_to_en.generate(**batch)
            english = tokenizer_fr_to_en.batch_decode(translation, skip_special_tokens=True)[0]

            current['aug_context'] = english

        previous_aug_passage = current['aug_context']
        previous_passage = passage
        aug_train[qid] = current

        print("Original:")
        print(passage)
        print("Augmented:")
        print(english)
        print("Answers:")
        print(answers)
        
    with open(args.predictions_save_path + "augmented_squad2_sec1.json", 'w') as fp:
        json.dump(aug_train, fp)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)