#! /usr/bin/env python

import argparse
import os
import sys
import json

import torch
import numpy as np
from scipy.stats import entropy

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


from models import ElectraQA

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
parser.add_argument('--predictions_save_path', type=str, help="Where to save predicted values")
parser.add_argument('--model_type', type=str, default="electra", help="Where to save predicted values")

def strip_accents(text):
    text = text.replace("ö", "o").replace("ü", "u").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n").replace("ç", "c").replace("â", "a").replace("ê", "e").replace("î", "i").replace("ô", "o").replace("û", "u").replace("à", "a").replace("è", "e").replace("ì", "i").replace("ò", "o").replace("ù", "u").replace("ë", "e").replace("ï", "i").replace("ä", "a").replace("ć", "c").replace("ń", "n").replace("ś", "s").replace("ź", "z").replace("ł", "l").replace("ż", "z").replace("ą", "a").replace("ę", "e").replace("š", "s").replace("ř", "r").replace("ů", "u").replace("č", "c").replace("ě", "e").replace("ž", "z").replace("ý", "y").replace("ā", "a").replace("ē", "e").replace("ī", "i").replace("ō", "o").replace("ū", "u").replace("õ", "o").replace("\u1ea1", "a").replace("\u1eb1", "a").replace("\u1ec7", "e")
    text = text.replace("å", "a")
    return text


def sig(x):
    return 1/(1+np.exp(-x))


def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    dev_data = load_dataset('squad_v2', split='validation')
    print(len(dev_data))

    electrasquad2 = "ahotrod/electra_large_discriminator_squad2_512"
    electrasquad1 = "mrm8488/electra-large-finetuned-squadv1"
    huggingface_model = electrasquad1
    if args.model_type == "bert":
        huggingface_model = "deepset/bert-large-uncased-whole-word-masking-squad2"
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
    model = AutoModelForQuestionAnswering.from_pretrained(huggingface_model)
    count = 0
    span_predictions = {}
    # entropy_on = []
    # entropy_off = []
    pred_start_probs = []
    pred_end_probs = []
    for ex in dev_data:
        count+=1
        print(count)
        # if count<3105:
        #     continue
        # if count==2:
        #    break
        question, passage, qid = ex["question"], ex["context"].replace("\n", ""), ex["id"]
        inputs = tokenizer.encode_plus(question, passage, add_special_tokens=True, return_tensors="pt")        
        inp_ids = inputs["input_ids"]
        if inp_ids.shape[1] > 512:
            # print("in here")
            inputs["input_ids"] = inputs["input_ids"][:,:512]
            inp_ids = inp_ids[:,:512]
        # start_logits, end_logits = model(**inputs)
        start_logits, end_logits = model(input_ids=inp_ids)
        answer_start = torch.argmax(start_logits)
        answer_end = torch.argmax(end_logits)
        inp_ids = inputs["input_ids"].tolist()[0]
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inp_ids[answer_start:answer_end+1]))

        answer = answer.replace(" ' ", "'").replace(" - ", "-").replace(" ( ", " (").replace("( ", "(").replace(" ) ", ") ").replace(" )", ")").replace(" , ", ", ").replace(" ? ", "? ").replace(" . ", ". ").replace(" % ", "% ").replace(" %", "%").replace(" $ ", " $").replace("$ ", "$").replace(" £ ", " £").replace("£ ", "£").replace(" / ", "/").replace(" [ ", "[").replace("[ ", "[").replace(" ] ", "] ").replace(" ]", "]").replace(" !", "!").replace("* ", "*")
        for p in range(10):
            for q in range(10):
                answer = answer.replace(str(p)+", "+str(q), str(p)+","+str(q))
                answer = answer.replace(str(p)+". "+str(q), str(p)+"."+str(q))

        answer = answer.replace("k ( n", "k(n").replace(" :", ":").replace(" ;", ";").replace("~ ", "~").replace("# ", "#").replace(" (n)", "(n)").replace(" (n2)", "(n2)").replace(" (n3)", "(n3)")
        if "\u00b0" in answer:
            answer = answer.replace("\u00b0 ", "\u00b0")
        if "\u2019" in answer:
            answer = answer.replace(" \u2019 ", "\u2019")
        if "\u2212" in answer:
            pass
            #answer = answer.replace("\u2212 f", "\u2212f")
        #if "\"" in answer:
        #    answer = answer.replace("\" ", "tochangeback", 1).replace(" \"", "\"").replace("tochangeback", "\"")
            #print(answer)
        if "\u2013" in answer:
            answer = answer.replace(" \u2013 ", "\u2013")

        answer = answer.replace("s'", "s' ").replace(" o 2", " o2").replace(" n o2", " no2").replace(" o 3", " o3").replace(" \u2032 ", "\u2032")
        quotations_pos = []
        if '"' in answer:
            quotations_pos = [x for x in range(len(answer)) if answer.startswith('\u0022', x)]
        quotations_pos.reverse()
        if len(quotations_pos) > 0 and len(quotations_pos) % 2 ==0:
            for k, qu in enumerate(quotations_pos):
                if k%2 == 1:
                    answer = answer[:qu+1] + answer[qu+2:]
                else:
                    answer = answer[:qu-1] + answer[qu:]

        if answer!="":
            noSpaceToWithSpace = []
            for pos, char in enumerate(passage):
                if char!=' ':
                    #if char == 'ö':
                    #    print("got one")
                    noSpaceToWithSpace.append(pos)
            passage_noSpace = strip_accents(passage.lower().replace(" ", ""))
            answer_noSpace = strip_accents(answer.replace(" ", ""))
            if answer_noSpace in passage_noSpace:
                start_char_noSpace = passage_noSpace.find(answer_noSpace)
                end_char_noSpace = start_char_noSpace + len(answer_noSpace) - 1
                start_char = noSpaceToWithSpace[start_char_noSpace]
                end_char = noSpaceToWithSpace[end_char_noSpace]
                answer = passage[start_char:end_char+1]
            else:
                #pass
                #if answer not in passage.lower():
                print(passage.lower())
                print(answer)
                print(' ')

        span_predictions[qid] = answer


    with open(args.predictions_save_path + "predictions.json", 'w') as fp:
        json.dump(span_predictions, fp)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)