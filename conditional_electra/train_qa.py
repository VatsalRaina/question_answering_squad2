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
from transformers import get_linear_schedule_with_warmup

from models import ElectraQAExtension

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
parser.add_argument('--learning_rate', type=float, default=3e-5, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=1, help='Specify the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--save_path', type=str, help='Load path to which trained model will be saved')
parser.add_argument('--model_path', type=str, help='Load path to a trained model that will be finetuned')

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
    return -1,0

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Set the seed value all over the place to make this reproducible.
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Choose device
    device = get_default_device()

    train_data = load_dataset('squad_v2', split='train')
    print(train_data[1953])

    electra_base = "google/electra-base-discriminator"
    electra_large = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)


    start_positions_true = []
    end_positions_true = []
    labels = []
    input_ids = []
    token_type_ids = []
    count = 0
    no_answer = 0
    too_long = 0
    for ex in train_data:
        count+=1
        # if count==64:
        #    break
        question, passage = ex["question"], ex["context"]
        combo = question + " [SEP] " + passage
        inp_ids = tokenizer.encode(combo)
        if len(inp_ids) > 512:
            too_long+=1
            inp_ids = inp_ids[:512]
        tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]  # Indicates whether part of sentence A or B -> 102 is Id of [SEP] token
        #print(ex)
        if len(ex["answers"]["text"])==0:
            start_idx, end_idx = 0, 0
            lab = 0
        else:
            ans_ids = tokenizer.encode(ex["answers"]["text"][0])
            ans_ids = ans_ids[1:-1]  # Remove the automatically added [CLS] and [SEP] tokens at beginning and end
            start_idx, end_idx = _find_sub_list(ans_ids, inp_ids)
            if start_idx == -1:
                continue
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
    labels = torch.tensor(labels)
    labels = labels.float().to(device)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.long().to(device)
    attention_masks = torch.tensor(attention_masks)
    attention_masks = attention_masks.long().to(device)
    # Create the DataLoader for training set.
    train_data = TensorDataset(input_ids, start_positions_true, end_positions_true, token_type_ids, attention_masks, labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    model = ElectraQAExtension(args.model_path, device).to(device)
    # Set weights from model_independent and also fix those paramaters so that they can't be trained further.

    parameters = []
    parameters.extend(model.conditional_layer1.parameters())
    parameters.extend(model.conditional_layer2.parameters())
    parameters.extend(model.qa_start.parameters())
    parameters.extend(model.qa_end.parameters())
    parameters.extend(model.classifier.parameters())
    optimizer = AdamW(parameters,
                    lr = args.learning_rate,
                    eps = args.adam_epsilon
                    # weight_decay = 0.01
                    )

    loss_values = []

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * args.n_epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 306,
                                                num_training_steps = total_steps)


    # accumulation_steps = 16

    criterion_verification = torch.nn.BCELoss()
    criterion_qa = torch.nn.CrossEntropyLoss()

    for epoch in range(args.n_epochs):
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.n_epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()
        model.zero_grad()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_start_pos_true = batch[1].to(device)
            b_end_pos_true = batch[2].to(device)
            b_tok_typ_ids = batch[3].to(device)
            b_att_msks = batch[4].to(device)
            b_labs = batch[5].to(device)
            model.zero_grad()
            start_logits, end_logits, verification_logits = model(input_ids=b_input_ids, attention_mask=b_att_msks, token_type_ids=b_tok_typ_ids)
            
            loss_verification = criterion_verification(verification_logits, b_labs)
            loss_start = criterion_qa(start_logits, b_start_pos_true)
            loss_end = criterion_qa(end_logits, b_end_pos_true)

            loss_qa = (loss_start + loss_end) / 2

            alpha1 = 0.5
            alpha2 = 1.0 - alpha1

            loss = alpha1*loss_verification + alpha2*loss_qa
            total_loss += loss.item()
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # if (step+1) % accumulation_steps == 0:
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
                # model.zero_grad()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    # Save the model to a file
    file_path = args.save_path+'electra_conditionalQA_seed'+str(args.seed)+'.pt'
    torch.save(model, file_path)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)